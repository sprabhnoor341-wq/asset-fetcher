"""
main.py — AI Asset Fetcher with file‑based job persistence, global queue,
and WebSocket real‑time updates.
"""

import asyncio
import os
import re
import shutil
import zipfile
import uuid
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import quote, urlencode
import random
import time

import aiohttp
import requests
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from llm_engine import (
    prepare_model,
    generate_3d,
    generate_audio,
    generate_textures,
    generate_ui,
    generate_script_architecture,
    generate_single_script,
)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
SKETCHFAB_TOKEN = "1d08911549ac417ba68839d196823657"
FREESOUND_API_KEY = "iGPvCX6Wh4WOLrhLomGhEvQHDRoB0JNmB4FrUwzJ"
SKETCHFAB_TOKEN = os.getenv("SKETCHFAB_TOKEN", SKETCHFAB_TOKEN)
FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY", FREESOUND_API_KEY)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Initializing NVIDIA Nemotron client...")
    ready = await asyncio.to_thread(prepare_model)
    if ready:
        print("[startup] NVIDIA Nemotron client ready.")
    else:
        print("[startup] NVIDIA Nemotron unavailable.")
    
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(global_worker())
    print("[startup] Global queue worker started.")
    
    yield
    print("[shutdown] Application shutting down.")


app = FastAPI(title="AI Asset Fetcher", lifespan=lifespan)

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
DOWNLOAD_DIR = BASE_DIR / "static" / "downloads"
JOBS_DIR = BASE_DIR / "jobs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ---------------------------------------------------------------------------
# Global Queue & Job Storage
# ---------------------------------------------------------------------------
job_queue: asyncio.Queue = asyncio.Queue()
global_semaphore = asyncio.Semaphore(1)

active_jobs: Dict[str, dict] = {}
job_subscribers: Dict[str, List[WebSocket]] = {}

MAX_CONCURRENT = 2
SCRIPT_CONCURRENCY = 3

SKETCHFAB_API = "https://api.sketchfab.com/v3"
FREESOUND_API = "https://freesound.org/apiv2"
POLYHAVEN_API = "https://api.polyhaven.com"
AMBIENTCG_API = "https://ambientcg.com/api/v1"
ICONIFY_API = "https://api.iconify.design"
USER_AGENT = "AssetDrop/1.0"


# ---------------------------------------------------------------------------
# Job File Helpers
# ---------------------------------------------------------------------------
def get_job_file_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def load_job_state(job_id: str) -> Optional[dict]:
    path = get_job_file_path(job_id)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


def save_job_state(job_id: str, updates: dict):
    path = get_job_file_path(job_id)
    current = load_job_state(job_id) or {}
    current.update(updates)
    current["updated_at"] = time.time()
    with open(path, "w") as f:
        json.dump(current, f, indent=2)


def create_job_file(job_id: str, prompt: str, engine: str):
    initial = {
        "job_id": job_id,
        "prompt": prompt,
        "engine": engine,
        "status": "queued",
        "logs": [],
        "progress": {"percent": 0, "current": 0, "total": 0},
        "result_url": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "error": None,
    }
    save_job_state(job_id, initial)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._ ")
    return name[:120] or "assets"


def _sketchfab_headers() -> dict:
    return {"Authorization": f"Token {SKETCHFAB_TOKEN}"} if SKETCHFAB_TOKEN else {}


def _freesound_headers() -> dict:
    return {"Authorization": f"Token {FREESOUND_API_KEY}"} if FREESOUND_API_KEY else {}


async def download_to_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    headers: Optional[dict] = None,
    timeout: int = 120,
) -> Optional[Path]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with session.get(
        url, headers=headers or {}, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        content_type = resp.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            return None
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        with open(dest, "wb") as f:
            async for chunk in resp.content.iter_chunked(8192):
                f.write(chunk)
        return dest


async def download_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    headers: Optional[dict] = None,
    timeout: int = 120,
    retries: int = 3,
    delay: int = 5,
) -> Optional[Path]:
    for attempt in range(retries + 1):
        try:
            result = await download_to_file(session, url, dest, headers, timeout)
            if result is None:
                return None
            return result
        except RuntimeError as e:
            if "HTTP 429" in str(e):
                if attempt < retries:
                    await asyncio.sleep(delay)
                    continue
                else:
                    return None
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return None
    return None


def folder_is_empty(folder: Path) -> bool:
    if not folder.exists():
        return True
    return not any(folder.rglob("*"))


# ---------------------------------------------------------------------------
# Term Sanitization & Smart Search
# ---------------------------------------------------------------------------
def sanitize_term(term: str) -> str:
    suffixes = [
        '_3d_model', '_model3d', '_3d_asset', '_3d', '_model', '_asset',
        '_interior', '_exterior', '_pack', '_variant', '_lowpoly', '_highpoly',
        '_game_ready', '_webgl', '_gltf', '_fbx', '_obj',
        '_sound', '_audio', '_sfx', '_effect',
        '_texture', '_material', '_map', '_tile',
        '_ui', '_icon', '_button', '_element'
    ]
    cleaned = term
    for suffix in suffixes:
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    cleaned = cleaned.rstrip('_').rstrip('-')
    return cleaned if cleaned else term


def generate_search_variants(term: str) -> List[str]:
    term = sanitize_term(term)
    variants = [term]
    variants.append(term.replace('_', ' '))
    prefixes = ['small_', 'modern_', 'stylized_', 'lowpoly_', 'simple_', 'generic_']
    simplified = term
    for prefix in prefixes:
        if simplified.startswith(prefix):
            simplified = simplified[len(prefix):]
            break
    if simplified != term and simplified:
        variants.append(simplified)
    if term.endswith('es') and len(term) > 2:
        variants.append(term[:-2] + 'h')
    elif term.endswith('s') and not term.endswith('ss'):
        variants.append(term[:-1])
    variants = [v.strip() for v in variants if v.strip()]
    return list(dict.fromkeys(variants))


# ---------------------------------------------------------------------------
# ZIP Extraction
# ---------------------------------------------------------------------------
def extract_and_organize_3d(zip_path: Path, term_folder: Path) -> bool:
    if not zip_path.exists():
        return False
    term_folder.mkdir(parents=True, exist_ok=True)
    temp_dir = term_folder / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            zip_path.unlink()
        except OSError:
            pass
        return False
    try:
        zip_path.unlink()
    except OSError:
        pass
    model_dir = term_folder / "model"
    textures_dir = term_folder / "textures"
    model_dir.mkdir(exist_ok=True)
    textures_dir.mkdir(exist_ok=True)
    mesh_extensions = {'.fbx', '.glb', '.gltf', '.obj', '.blend', '.dae', '.3ds', '.stl', '.ply'}
    image_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.tif', '.bmp', '.psd', '.hdr', '.exr'}
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            src = Path(root) / file
            ext = src.suffix.lower()
            if ext in mesh_extensions:
                dest = model_dir / file
                shutil.move(str(src), str(dest))
            elif ext in image_extensions:
                dest = textures_dir / file
                shutil.move(str(src), str(dest))
            else:
                dest = term_folder / file
                shutil.move(str(src), str(dest))
    shutil.rmtree(temp_dir, ignore_errors=True)
    for nested_zip in list(term_folder.rglob("*.zip")) + list(term_folder.rglob("*.ZIP")):
        nested_temp = term_folder / "_nested_temp"
        nested_temp.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(nested_zip, "r") as zf:
                zf.extractall(nested_temp)
            nested_zip.unlink()
            for root, dirs, files in os.walk(nested_temp):
                for file in files:
                    src = Path(root) / file
                    ext = src.suffix.lower()
                    if ext in mesh_extensions:
                        dest = model_dir / file
                        shutil.move(str(src), str(dest))
                    elif ext in image_extensions:
                        dest = textures_dir / file
                        shutil.move(str(src), str(dest))
                    else:
                        dest = term_folder / file
                        shutil.move(str(src), str(dest))
            shutil.rmtree(nested_temp, ignore_errors=True)
        except Exception:
            shutil.rmtree(nested_temp, ignore_errors=True)
            continue
    license_path = term_folder / "license.txt"
    license_text = (
        "Asset downloaded via Sketchfab/PolyHaven API.\n"
        "Please check the original source for specific license details.\n"
        "Sketchfab: https://sketchfab.com/\n"
        "PolyHaven: https://polyhaven.com/"
    )
    license_path.write_text(license_text)
    return True


def extract_and_cleanup_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    except Exception:
        try:
            zip_path.unlink()
        except OSError:
            pass
        return
    try:
        zip_path.unlink()
    except OSError:
        pass
    for f in list(extract_to.rglob("*")):
        if f.is_file() and f.parent != extract_to:
            dest = extract_to / f.name
            if not dest.exists():
                try:
                    f.rename(dest)
                except OSError:
                    pass
    for d in sorted(extract_to.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Asset Fetchers
# ---------------------------------------------------------------------------
async def _search_sketchfab(session: aiohttp.ClientSession, term: str) -> Optional[tuple]:
    search_url = f"{SKETCHFAB_API}/search"
    params = {"type": "models", "downloadable": "true", "q": term, "count": "10"}
    headers = _sketchfab_headers()
    for attempt in range(4):
        try:
            async with session.get(
                search_url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    if attempt < 3:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return None
                if resp.status != 200:
                    return None
                data = await resp.json()
                results = data.get("results", [])
                if not results:
                    return None
                uid = None
                for r in results:
                    if r.get("isDownloadable"):
                        uid = r.get("uid")
                        break
                if uid is None and results:
                    uid = results[0].get("uid")
                if not uid:
                    return None
                return data, uid
        except Exception:
            if attempt == 3:
                return None
            await asyncio.sleep(2)
    return None


async def fetch_sketchfab(session: aiohttp.ClientSession, term: str, assets_dir: Path, sem: asyncio.Semaphore) -> bool:
    target = assets_dir / term
    target.mkdir(parents=True, exist_ok=True)
    async with sem:
        variants = generate_search_variants(term)
        for variant in variants:
            result = await _search_sketchfab(session, variant)
            if result is not None:
                data, uid = result
                if variant != term:
                    print(f"  [Retry] '{term}' matched as '{variant}'")
                dl_endpoint = f"{SKETCHFAB_API}/models/{uid}/download"
                headers = _sketchfab_headers()
                for attempt in range(4):
                    try:
                        async with session.get(
                            dl_endpoint,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 429:
                                if attempt < 3:
                                    await asyncio.sleep(5)
                                    continue
                                else:
                                    return False
                            if resp.status != 200:
                                return False
                            dl_data = await resp.json()
                            break
                    except Exception:
                        if attempt == 3:
                            return False
                        await asyncio.sleep(2)
                else:
                    return False
                dl_url = _extract_sketchfab_url(dl_data)
                if not dl_url:
                    return False
                zip_path = assets_dir / f"sketchfab_{uid}.zip"
                result_dl = await download_with_retry(session, dl_url, zip_path, headers=None, retries=3, delay=5)
                if result_dl is None:
                    return False
                success = extract_and_organize_3d(zip_path, target)
                if success:
                    print(f"  [Sketchfab] OK '{term}'")
                return success
        print(f"  [skip] '{term}': No results found")
        return False


def _extract_sketchfab_url(dl_data) -> Optional[str]:
    def candidate(fmt):
        if isinstance(fmt, dict):
            return fmt.get("url") or fmt.get("download_url")
        return None
    def fmt(fmt):
        if isinstance(fmt, dict):
            return (fmt.get("format") or "").lower()
        return ""
    candidates = []
    if isinstance(dl_data, list):
        candidates = dl_data
    elif isinstance(dl_data, dict):
        if "formats" in dl_data and isinstance(dl_data["formats"], list):
            candidates = dl_data["formats"]
        elif isinstance(next(iter(dl_data.values()), None), dict):
            for key, val in dl_data.items():
                if isinstance(val, dict):
                    candidates.append({"format": key, **val})
        else:
            return dl_data.get("url")
    for pref in ("gltf", "glb", ""):
        for c in candidates:
            if pref and pref not in fmt(c):
                continue
            u = candidate(c)
            if u:
                return u
    return None


async def _search_polyhaven(session: aiohttp.ClientSession, term: str) -> Optional[str]:
    search_url = f"{POLYHAVEN_API}/assets?{urlencode({'query': term})}"
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(3):
        try:
            async with session.get(
                search_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return None
                if resp.status != 200:
                    return None
                data = await resp.json()
                if not isinstance(data, dict) or not data:
                    return None
                term_lower = term.lower()
                for aid, meta in data.items():
                    if not isinstance(meta, dict):
                        continue
                    name = (meta.get("name") or "").lower()
                    tags = " ".join(meta.get("tags") or []).lower()
                    if term_lower in name or term_lower in tags:
                        return aid
                return None
        except Exception:
            if attempt == 2:
                return None
            await asyncio.sleep(2)
    return None


async def fetch_polyhaven(session: aiohttp.ClientSession, term: str, assets_dir: Path, sem: asyncio.Semaphore) -> bool:
    is_3d = "3D" in str(assets_dir)
    if is_3d:
        target = assets_dir / term
        target.mkdir(parents=True, exist_ok=True)
    else:
        target = assets_dir
        target.mkdir(parents=True, exist_ok=True)
    async with sem:
        variants = generate_search_variants(term)
        for variant in variants:
            asset_id = await _search_polyhaven(session, variant)
            if asset_id:
                if variant != term:
                    print(f"  [Retry] '{term}' matched as '{variant}'")
                files_url = f"{POLYHAVEN_API}/files/{quote(asset_id)}"
                headers = {"User-Agent": USER_AGENT}
                for attempt in range(3):
                    try:
                        async with session.get(
                            files_url,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 429:
                                if attempt < 2:
                                    await asyncio.sleep(5)
                                    continue
                                else:
                                    return False
                            if resp.status != 200:
                                return False
                            files = await resp.json()
                            break
                    except Exception:
                        if attempt == 2:
                            return False
                        await asyncio.sleep(2)
                else:
                    return False
                url = _find_polyhaven_url(files)
                if not url:
                    return False
                ext = Path(url.split("?")[0]).suffix or ".zip"
                dest = assets_dir / f"polyhaven_{asset_id}{ext[:8]}"
                result_dl = await download_with_retry(session, url, dest, headers=headers, retries=3, delay=5)
                if result_dl is None:
                    return False
                if dest.suffix.lower() == '.zip':
                    if is_3d:
                        success = extract_and_organize_3d(dest, target)
                    else:
                        extract_and_cleanup_zip(dest, target)
                        success = True
                else:
                    if is_3d:
                        model_dir = target / "model"
                        model_dir.mkdir(exist_ok=True)
                        shutil.move(str(dest), str(model_dir / dest.name))
                        license_path = target / "license.txt"
                        license_path.write_text("Asset downloaded via PolyHaven API. Please check the original source for license details.\nhttps://polyhaven.com/")
                        success = True
                    else:
                        shutil.move(str(dest), str(target / dest.name))
                        success = True
                if success:
                    if is_3d:
                        print(f"  [PolyHaven] OK '{term}' (3D)")
                    else:
                        print(f"  [PolyHaven] OK '{term}'")
                return success
        print(f"  [skip] '{term}': No results found")
        return False


def _find_polyhaven_url(files) -> Optional[str]:
    best = None
    def _key_to_num(k: str) -> int:
        try:
            return int(k[0])
        except Exception:
            return 2
    def walk(node):
        nonlocal best
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    if k in ("1k", "2k", "4k", "8k") and isinstance(v.get("url"), str):
                        if best is None or _key_to_num(k) > _key_to_num(best[0]):
                            best = (k, v["url"])
                    walk(v)
                elif k == "url" and isinstance(v, str) and v.startswith("http"):
                    best = ("2k", v)
    walk(files)
    return best[1] if best else None


async def _search_ambientcg(session: aiohttp.ClientSession, term: str) -> Optional[str]:
    phrase = quote(term)
    search_url = f"{AMBIENTCG_API}/json?method=Search&phrase={phrase}"
    for attempt in range(3):
        try:
            async with session.get(
                search_url,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return None
                if resp.status != 200:
                    return None
                data = await resp.json()
                assets = data.get("foundAssets", [])
                if not assets:
                    return None
                asset_id = assets[0].get("assetId")
                return asset_id
        except Exception:
            if attempt == 2:
                return None
            await asyncio.sleep(2)
    return None


async def fetch_ambientcg(session: aiohttp.ClientSession, term: str, assets_dir: Path, sem: asyncio.Semaphore) -> bool:
    target = assets_dir
    target.mkdir(parents=True, exist_ok=True)
    async with sem:
        variants = generate_search_variants(term)
        for variant in variants:
            asset_id = await _search_ambientcg(session, variant)
            if asset_id:
                if variant != term:
                    print(f"  [Retry] '{term}' matched as '{variant}'")
                dl_params = {"method": "download", "assetId": asset_id}
                for attempt in range(3):
                    try:
                        async with session.get(
                            f"{AMBIENTCG_API}/json",
                            params=dl_params,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 429:
                                if attempt < 2:
                                    await asyncio.sleep(5)
                                    continue
                                else:
                                    return False
                            if resp.status != 200:
                                return False
                            dl_data = await resp.json()
                            break
                    except Exception:
                        if attempt == 2:
                            return False
                        await asyncio.sleep(2)
                else:
                    return False
                dl_info = dl_data.get("foundDownloadInfo") or {}
                dl_url = dl_info.get("downloadUrl")
                if not dl_url:
                    return False
                zip_path = target / f"ambientcg_{asset_id}.zip"
                result_dl = await download_with_retry(session, dl_url, zip_path, headers=None, retries=3, delay=5)
                if result_dl is None:
                    return False
                extract_and_cleanup_zip(zip_path, target)
                print(f"  [AmbientCG] OK '{term}'")
                return True
        print(f"  [skip] '{term}': No results found")
        return False


async def _search_freesound(session: aiohttp.ClientSession, term: str) -> Optional[tuple]:
    search_url = f"{FREESOUND_API}/search/text/"
    params = {
        "query": term,
        "filter": "duration:[0 TO 10]",
        "sort": "score_desc",
        "fields": "id,name,previews",
        "page_size": "1",
    }
    headers = _freesound_headers()
    for attempt in range(4):
        try:
            async with session.get(
                search_url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 429:
                    if attempt < 3:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return None
                if resp.status != 200:
                    return None
                data = await resp.json()
                results = data.get("results", [])
                if not results:
                    return None
                top = results[0]
                previews = top.get("previews") or {}
                url = previews.get("preview-hq-mp3")
                if not url:
                    return None
                sound_id = top.get("id", 0)
                return url, sound_id
        except Exception:
            if attempt == 3:
                return None
            await asyncio.sleep(2)
    return None


async def fetch_freesound(session: aiohttp.ClientSession, term: str, assets_dir: Path, sem: asyncio.Semaphore) -> bool:
    target = assets_dir / "audio"
    target.mkdir(parents=True, exist_ok=True)
    async with sem:
        variants = generate_search_variants(term)
        for variant in variants:
            result = await _search_freesound(session, variant)
            if result is not None:
                url, sound_id = result
                if variant != term:
                    print(f"  [Retry] '{term}' matched as '{variant}'")
                dest = target / f"sound_{sound_id}.mp3"
                result_dl = await download_with_retry(session, url, dest, headers=None, retries=3, delay=5)
                if result_dl is None:
                    return False
                print(f"  [Freesound] OK '{term}'")
                return True
        print(f"  [skip] '{term}': No results found")
        return False


async def fetch_iconify(session: aiohttp.ClientSession, term: str, assets_dir: Path, sem: asyncio.Semaphore) -> bool:
    target = assets_dir / "ui"
    target.mkdir(parents=True, exist_ok=True)
    clean = sanitize_term(term).strip().lower()
    icon_name = clean.replace('_', '-').replace(' ', '-')
    icon_name = re.sub(r'-+', '-', icon_name).strip('-')
    if not icon_name:
        print(f"  [skip] '{term}': Empty term after sanitization")
        return False
    url = f"{ICONIFY_API}/game-icons/{icon_name}.svg?width=128&height=128&color=white"
    dest = target / f"game-icons_{icon_name}.svg"
    async with sem:
        result = await download_with_retry(session, url, dest, headers=None, timeout=30, retries=2, delay=3)
        if result is not None:
            print(f"  [Iconify] OK '{term}' (game-icons/{icon_name})")
            return True
        print(f"  [skip] '{term}': Icon not found")
        return False


# ---------------------------------------------------------------------------
# Asset Worker
# ---------------------------------------------------------------------------
async def process_task(session: aiohttp.ClientSession, task: dict, cache_dir: Path, sem: asyncio.Semaphore) -> bool:
    category = task["category"]
    term = task["term"]
    try:
        if category == "3D":
            if await fetch_sketchfab(session, term, cache_dir / "3D", sem):
                return True
            return await fetch_polyhaven(session, term, cache_dir / "3D", sem)
        elif category == "audio":
            return await fetch_freesound(session, term, cache_dir, sem)
        elif category == "textures":
            if await fetch_polyhaven(session, term, cache_dir / "textures", sem):
                return True
            return await fetch_ambientcg(session, term, cache_dir / "textures", sem)
        elif category == "ui":
            return await fetch_iconify(session, term, cache_dir, sem)
        else:
            return False
    except Exception:
        return False


async def fetch_all_tasks(tasks: List[Dict], cache_dir: Path) -> int:
    random.shuffle(tasks)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        worker_tasks = [
            asyncio.create_task(process_task(session, task, cache_dir, sem))
            for task in tasks
        ]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)
    return sum(1 for r in results if r is True)


# ---------------------------------------------------------------------------
# Script Generation Orchestration
# ---------------------------------------------------------------------------
async def generate_and_write_scripts(prompt: str, engine: str, scripts_dir: Path) -> int:
    print(f"[Script] Architect planning for '{prompt}' using {engine}...")
    file_paths = await generate_script_architecture(prompt, engine)
    if not file_paths:
        print("[Script] No files returned from Architect.")
        return 0

    print(f"[Script] Architect planned {len(file_paths)} files.")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(SCRIPT_CONCURRENCY)

    async def write_one(file_path: str) -> bool:
        async with sem:
            full_path = scripts_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Script] Generating {file_path}...")
            code = await generate_single_script(prompt, engine, file_path)
            if not code:
                print(f"[Script] Failed to generate content for {file_path}")
                return False
            try:
                full_path.write_text(code, encoding="utf-8")
                print(f"[Script] ✓ {file_path} written.")
                return True
            except Exception as e:
                print(f"[Script] Error writing {file_path}: {e}")
                return False

    tasks = [write_one(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return sum(1 for r in results if r is True)


# ---------------------------------------------------------------------------
# Broadcast helper (fix: ensure progress is a dict)
# ---------------------------------------------------------------------------
async def broadcast_job_update(job_id: str, message: dict):
    if job_id in active_jobs:
        if message.get("type") == "log":
            active_jobs[job_id].setdefault("logs", []).append(message.get("message"))
        elif message.get("type") == "progress":
            # Ensure progress is a dict
            if "progress" not in active_jobs[job_id] or not isinstance(active_jobs[job_id]["progress"], dict):
                active_jobs[job_id]["progress"] = {"percent": 0, "current": 0, "total": 0}
            active_jobs[job_id]["progress"]["percent"] = message.get("percent", 0)
            active_jobs[job_id]["progress"]["current"] = message.get("current", 0)
            active_jobs[job_id]["progress"]["total"] = message.get("total", 0)
            active_jobs[job_id]["message"] = message.get("message", "")
        elif message.get("type") == "done":
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["result_url"] = message.get("url")
        elif message.get("type") == "error":
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = message.get("message")
        save_job_state(job_id, active_jobs[job_id])
    
    if job_id in job_subscribers:
        for ws in job_subscribers[job_id]:
            try:
                await ws.send_json(message)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Global Worker
# ---------------------------------------------------------------------------
async def global_worker():
    while True:
        job = await job_queue.get()
        try:
            async with global_semaphore:
                await process_job(job)
        except Exception as e:
            job_id = job.get("job_id")
            if job_id in active_jobs:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["error"] = str(e)
                save_job_state(job_id, active_jobs[job_id])
                await broadcast_job_update(job_id, {"type": "error", "message": str(e)})
            print(f"[Worker] Job {job_id} failed: {e}")
        finally:
            job_queue.task_done()


async def process_job(job: dict):
    job_id = job["job_id"]
    prompt = job["prompt"]
    engine = job["engine"]
    websocket = job.get("websocket")

    # Initialize in‑memory state (progress as dict)
    active_jobs[job_id] = {
        "status": "queued",
        "progress": {"percent": 0, "current": 0, "total": 0},
        "logs": [],
        "queued_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "result_url": None,
        "error": None,
        "message": "",
    }
    if websocket:
        job_subscribers.setdefault(job_id, []).append(websocket)

    create_job_file(job_id, prompt, engine)
    save_job_state(job_id, active_jobs[job_id])

    queue_position = job_queue.qsize()
    await broadcast_job_update(job_id, {"type": "queue", "position": queue_position})

    try:
        active_jobs[job_id]["started_at"] = time.time()
        active_jobs[job_id]["status"] = "processing"
        save_job_state(job_id, active_jobs[job_id])

        await broadcast_job_update(job_id, {"type": "log", "message": "Generating asset lists..."})
        tasks_3d, tasks_audio, tasks_textures, tasks_ui = await asyncio.gather(
            generate_3d(prompt, 10),
            generate_audio(prompt, 10),
            generate_textures(prompt, 10),
            generate_ui(prompt, 10),
        )
        all_tasks = []
        for term in tasks_3d:
            all_tasks.append({"category": "3D", "term": term})
        for term in tasks_audio:
            all_tasks.append({"category": "audio", "term": term})
        for term in tasks_textures:
            all_tasks.append({"category": "textures", "term": term})
        for term in tasks_ui:
            all_tasks.append({"category": "ui", "term": term})

        total_assets = len(all_tasks)
        await broadcast_job_update(job_id, {"type": "log", "message": f"Found {total_assets} assets to download."})

        safe_prompt = safe_name(prompt)
        cache_path = CACHE_DIR / safe_prompt
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
        for sub in ("3D", "audio", "textures", "ui", "scripts"):
            (cache_path / sub).mkdir(parents=True, exist_ok=True)

        await broadcast_job_update(job_id, {
            "type": "progress",
            "percent": 30,
            "current": 0,
            "total": total_assets,
            "message": "Downloading assets..."
        })
        fetch_count = await fetch_all_tasks(all_tasks, cache_path)
        await broadcast_job_update(job_id, {
            "type": "progress",
            "percent": 70,
            "current": fetch_count,
            "total": total_assets,
            "message": f"Downloaded {fetch_count} assets."
        })

        await broadcast_job_update(job_id, {"type": "log", "message": "Generating scripts..."})
        scripts_dir = cache_path / "scripts"
        scripts_success = await generate_and_write_scripts(prompt, engine, scripts_dir)
        await broadcast_job_update(job_id, {"type": "log", "message": f"Generated {scripts_success} scripts."})

        await broadcast_job_update(job_id, {"type": "log", "message": "Packaging ZIP..."})
        zip_path = DOWNLOAD_DIR / f"{safe_prompt}.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(
            base_name=str(DOWNLOAD_DIR / safe_prompt),
            format="zip",
            root_dir=str(cache_path),
        )
        shutil.rmtree(cache_path, ignore_errors=True)

        download_url = f"/static/downloads/{safe_prompt}.zip"
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["result_url"] = download_url
        active_jobs[job_id]["completed_at"] = time.time()
        save_job_state(job_id, active_jobs[job_id])
        await broadcast_job_update(job_id, {
            "type": "done",
            "url": download_url,
            "message": "Asset pack and scripts ready!"
        })

        print(f"[Worker] Job {job_id} completed.")

    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        save_job_state(job_id, active_jobs[job_id])
        await broadcast_job_update(job_id, {"type": "error", "message": str(e)})
        raise


# ---------------------------------------------------------------------------
# WebSocket: Generator (new or resume)
# ---------------------------------------------------------------------------
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    job_id = None
    await websocket.accept()
    try:
        query = websocket.query_params
        job_id_param = query.get("job_id")
        prompt = query.get("prompt", "").strip()
        engine = query.get("engine", "Unity")

        # ===== RESUME BRANCH =====
        if job_id_param:
            state = load_job_state(job_id_param)
            if not state:
                await websocket.send_json({"type": "error", "message": "Invalid Job ID"})
                await websocket.close()
                return

            job_id = job_id_param
            status = state.get("status")

            if status == "completed":
                # Replay logs
                for log in state.get("logs", []):
                    await websocket.send_json({"type": "log", "message": log})
                # Send progress (handle both dict and int legacy)
                prog = state.get("progress", {})
                if isinstance(prog, dict):
                    percent = prog.get("percent", 100)
                    current = prog.get("current", 0)
                    total = prog.get("total", 0)
                else:
                    # Legacy: progress is an integer (percent)
                    percent = prog if isinstance(prog, int) else 100
                    current = 0
                    total = 0
                await websocket.send_json({
                    "type": "progress",
                    "percent": percent,
                    "current": current,
                    "total": total,
                    "message": "Done"
                })
                await websocket.send_json({
                    "type": "done",
                    "url": state.get("result_url"),
                    "message": "Asset pack ready!"
                })
                await websocket.close()
                return

            elif status == "failed":
                await websocket.send_json({"type": "error", "message": state.get("error", "Job failed")})
                await websocket.close()
                return

            elif status in ("queued", "processing"):
                if job_id in active_jobs:
                    # Attach this websocket as a subscriber
                    job_subscribers.setdefault(job_id, []).append(websocket)
                    # Replay logs
                    for log in state.get("logs", []):
                        await websocket.send_json({"type": "log", "message": log})
                    # Send current progress
                    prog = state.get("progress", {})
                    if isinstance(prog, dict):
                        percent = prog.get("percent", 0)
                        current = prog.get("current", 0)
                        total = prog.get("total", 0)
                    else:
                        percent = prog if isinstance(prog, int) else 0
                        current = 0
                        total = 0
                    await websocket.send_json({
                        "type": "progress",
                        "percent": percent,
                        "current": current,
                        "total": total,
                        "message": state.get("message", "")
                    })
                    # Keep connection alive
                    while True:
                        try:
                            await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
                        except asyncio.TimeoutError:
                            try:
                                await websocket.send_json({"type": "ping"})
                            except:
                                break
                        except WebSocketDisconnect:
                            break
                else:
                    await websocket.send_json({"type": "error", "message": "Job was interrupted. Please start a new generation."})
                    await websocket.close()
                    return
            else:
                await websocket.send_json({"type": "error", "message": "Unknown job status"})
                await websocket.close()
                return

        # ===== NEW GENERATION BRANCH =====
        else:
            if not prompt:
                await websocket.send_json({"type": "error", "message": "Prompt is required."})
                await websocket.close()
                return

            job_id = secrets.token_hex(3)
            while get_job_file_path(job_id).exists():
                job_id = secrets.token_hex(3)

            await websocket.send_json({"type": "job_id", "id": job_id})
            create_job_file(job_id, prompt, engine)

            job = {
                "job_id": job_id,
                "prompt": prompt,
                "engine": engine,
                "websocket": websocket,
                "enqueued_at": time.time()
            }
            await job_queue.put(job)

            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "ping"})
                    except:
                        break
                except WebSocketDisconnect:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if job_id and job_id in job_subscribers:
            if websocket in job_subscribers.get(job_id, []):
                job_subscribers[job_id].remove(websocket)
            if not job_subscribers.get(job_id):
                del job_subscribers[job_id]
        try:
            await websocket.close()
        except:
            pass


# ---------------------------------------------------------------------------
# Other Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/fetch")
async def fetch_assets_api_deprecated():
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Please update your frontend to use the WebSocket endpoint at /ws/generate. See the new static/index.html for example usage."
    )


@app.post("/api/plan")
async def plan_assets(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    engine = body.get("engine", "Unity")
    qty = body.get("quantity", 10)
    qty_3d = body.get("qty_3d", qty)
    qty_audio = body.get("qty_audio", qty)
    qty_textures = body.get("qty_textures", qty)
    qty_ui = body.get("qty_ui", qty)

    tasks_3d, tasks_audio, tasks_textures, tasks_ui = await asyncio.gather(
        generate_3d(prompt, qty_3d),
        generate_audio(prompt, qty_audio),
        generate_textures(prompt, qty_textures),
        generate_ui(prompt, qty_ui),
    )
    return {
        "3D": tasks_3d,
        "audio": tasks_audio,
        "textures": tasks_textures,
        "ui": tasks_ui,
        "engine": engine,
    }


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    state = load_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=state)


@app.get("/api/search")
async def search_assets(q: str = "", count: int = 10):
    try:
        params = {"type": "models", "downloadable": "true", "q": q, "count": min(count, 24)}
        response = requests.get(
            f"{SKETCHFAB_API}/search",
            params=params,
            headers=_sketchfab_headers(),
            timeout=30
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        formatted = []
        for result in results:
            formatted.append({
                "uid": result.get("uid"),
                "name": result.get("name"),
                "description": result.get("description"),
                "downloadable": result.get("isDownloadable"),
                "thumbnails": result.get("thumbnails", {}).get("images", []),
            })
        return {"results": formatted}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)