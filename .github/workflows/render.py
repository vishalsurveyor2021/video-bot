"""Hybrid renderer: AI image -> (free HF GPU image-to-video clip, else zoom fallback) + Hindi voice + captions.

Input: env PAYLOAD (URL-encoded Groq JSON). Output: out/video.mp4, out/meta.json
Optional env: HF_TOKEN (Hugging Face token), HF_SPACES (comma separated Space ids), VOICE
"""
import asyncio
import glob
import json
import os
import random
import subprocess
import textwrap
import urllib.parse

import edge_tts
import httpx

W, H, FPS = 1080, 1920, 30
VOICE = os.getenv("VOICE", "hi-IN-MadhurNeural")  # female: hi-IN-SwaraNeural
HF_TOKEN = os.getenv("HF_TOKEN", "")
# Image-to-video Spaces tried in order. Names can disappear: change HF_SPACES if needed.
HF_SPACES = [s.strip() for s in os.getenv("HF_SPACES", "ChopperBlu/wan22-i2v,multimodalart/wan2-2-fp8da-aoti-preview").split(",") if s.strip()]
CLIP_TIMEOUT = int(os.getenv("CLIP_TIMEOUT", "300"))
IMAGE_URL = os.getenv(
    "IMAGE_URL_TEMPLATE",
    "https://image.pollinations.ai/prompt/{prompt}?width=1080&height=1920&model=flux&nologo=true&seed={seed}",
)
STYLE_SUFFIX = ", photorealistic live-action Hollywood movie still, ultra realistic, 4K HDR, volumetric lighting, realistic shadows, natural skin texture, shallow depth of field, dramatic atmosphere, epic scale, vertical 9:16, no text, no watermark"
SEED = random.randint(1, 10**6)  # same seed for all scenes => more consistent characters
OUT, WORK = "out", "work"


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-800:]}")


def find_font():
    for pat in ("/usr/share/fonts/**/NotoSansDevanagari-Bold.ttf", "/usr/share/fonts/**/NotoSansDevanagari*.ttf"):
        f = glob.glob(pat, recursive=True)
        if f:
            return f[0]
    raise RuntimeError("Devanagari font missing")


def duration(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(p.stdout.strip())


async def voice(text, out):
    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, VOICE).save(out)
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(3)


async def image(client, prompt, out):
    url = IMAGE_URL.format(prompt=urllib.parse.quote(prompt + STYLE_SUFFIX), seed=SEED)
    for _ in range(3):
        try:
            r = await client.get(url, timeout=90, follow_redirects=True)
            if r.status_code == 200 and len(r.content) > 5000 and r.content[:3] in (b"\xff\xd8\xff", b"\x89PN"):
                open(out, "wb").write(r.content)
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(3)
    print("image failed, using fallback background:", prompt[:60])
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x1b1b2f:s={W}x{H}", "-frames:v", "1", out])


# ---------------- AI image-to-video via free Hugging Face Spaces ----------------

def _find_video(obj):
    if isinstance(obj, str) and obj.lower().endswith((".mp4", ".webm", ".mov")) and os.path.exists(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = _find_video(v)
            if r:
                return r
    if isinstance(obj, (list, tuple)):
        for v in obj:
            r = _find_video(v)
            if r:
                return r
    return None


def ai_clip_from_space(space, img_path, prompt):
    """Auto-detects an image->video endpoint of a Gradio Space and calls it. Returns mp4 path or raises."""
    from gradio_client import Client, handle_file

    try:
        client = Client(space, hf_token=HF_TOKEN, verbose=False)
    except TypeError:
        client = Client(space, token=HF_TOKEN, verbose=False)
    info = client.view_api(return_format="dict", print_info=False)
    endpoints = info.get("named_endpoints", {})
    last_err = None
    for api_name, ep in endpoints.items():
        params = ep.get("parameters", [])
        comps = [p.get("component") for p in params]
        rets = [r.get("component") for r in ep.get("returns", [])]
        if "Image" not in comps or "Video" not in rets or comps.count("Image") != 1:
            continue
        kwargs, ok = {}, True
        for p in params:
            name, comp = p["parameter_name"], p.get("component")
            label = (p.get("label") or "").lower()
            if comp == "Image":
                kwargs[name] = handle_file(img_path)
            elif comp == "Textbox" and "negative" not in label and "negative" not in name:
                kwargs[name] = prompt
            elif p.get("parameter_has_default"):
                kwargs[name] = p["parameter_default"]
            else:
                ok = False
                break
        if not ok:
            continue
        try:
            print(f"  trying {space}{api_name}")
            job = client.submit(**kwargs, api_name=api_name)
            result = job.result(timeout=CLIP_TIMEOUT)
            path = _find_video(result)
            if path:
                return path
            last_err = "no video in result"
        except Exception as e:  # quota, timeout, runtime error...
            last_err = str(e)[:200]
    raise RuntimeError(last_err or "no image-to-video endpoint found")


def try_ai_clip(img_path, scene_prompt, idx):
    if not HF_TOKEN:
        return None
    motion = (scene_prompt[:350] + ", natural realistic motion, subtle camera push-in, cinematic, smooth")
    for space in HF_SPACES:
        try:
            src = ai_clip_from_space(space, img_path, motion)
            dst = f"{WORK}/ai{idx}.mp4"
            run(["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-an", "-pix_fmt", "yuv420p", dst])
            print(f"scene {idx}: AI clip OK via {space}")
            return dst
        except Exception as e:
            print(f"scene {idx}: {space} failed: {str(e)[:160]}")
    return None


# ---------------- clip assembly ----------------

def clip(img, audio, caption, font, out, idx, ai_clip=None):
    dur = duration(audio) + 0.4
    capf = f"{WORK}/cap{idx}.txt"
    open(capf, "w", encoding="utf-8").write(textwrap.fill(caption, width=22))
    if ai_clip:
        pad = max(dur - duration(ai_clip), 0) + 0.3
        base = (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,"
                f"fps={FPS},tpad=stop_mode=clone:stop_duration={pad:.2f}")
        inputs = ["-i", ai_clip, "-i", audio]
    else:
        frames = int(dur * FPS)
        zoom = "max(1.18-0.0007*on,1.0)" if idx % 2 else "min(1.0+0.0007*on,1.18)"
        base = (f"scale=1296:2304:force_original_aspect_ratio=increase,crop=1296:2304,"
                f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS}")
        inputs = ["-i", img, "-i", audio]
    vf = (
        f"{base},"
        f"drawbox=x=0:y=ih*0.60:w=iw:h=ih*0.28:color=black@0.45:t=fill,"
        f"drawtext=fontfile={font}:textfile={capf}:fontsize=62:fontcolor=white:borderw=3:bordercolor=black:"
        f"x=(w-text_w)/2:y=h*0.64:line_spacing=14,"
        f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(dur-0.3, 0):.2f}:d=0.3,format=yuv420p"
    )
    run(["ffmpeg", "-y", *inputs, "-map", "0:v", "-map", "1:a", "-vf", vf, "-af", "apad", "-t", f"{dur:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", out])


async def main():
       raw = os.environ["PAYLOAD"].strip()
   try:
       data = json.loads(raw)
   except ValueError:
       raw = urllib.parse.unquote_plus(raw).strip()
       raw = raw.removeprefix("```json").removesuffix("```").strip()
       data = json.loads(raw)
    scenes = data["scenes"][:8]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    font = find_font()

    async with httpx.AsyncClient() as client:
        tasks = []
        for i, s in enumerate(scenes):
            tasks.append(voice(s["scene_text"], f"{WORK}/a{i}.mp3"))
            tasks.append(image(client, s["image_prompt"], f"{WORK}/i{i}.jpg"))
        await asyncio.gather(*tasks)

    # AI video clips one by one (free GPU quota); stop trying after 2 failures in a row
    ai = {}
    fails = 0
    for i, s in enumerate(scenes):
        if fails >= 2:
            break
        path = await asyncio.to_thread(try_ai_clip, f"{WORK}/i{i}.jpg", s["image_prompt"], i)
        if path:
            ai[i], fails = path, 0
        else:
            fails += 1
    print(f"AI clips: {len(ai)}/{len(scenes)}")

    clips = []
    for i, s in enumerate(scenes):
        c = f"{WORK}/c{i}.mp4"
        clip(f"{WORK}/i{i}.jpg", f"{WORK}/a{i}.mp3", s["scene_text"], font, c, i, ai.get(i))
        clips.append(c)

    with open(f"{WORK}/list.txt", "w") as f:
        f.writelines(f"file '{os.path.abspath(c)}'\n" for c in clips)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", f"{WORK}/list.txt", "-c", "copy",
         "-movflags", "+faststart", f"{OUT}/video.mp4"])

    json.dump({"title": (data.get("title") or "Video")[:100], "description": data.get("description", "")},
              open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("done", os.path.getsize(f"{OUT}/video.mp4") // 1024, "KB")


asyncio.run(main())
