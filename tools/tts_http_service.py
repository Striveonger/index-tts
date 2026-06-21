import io
import os
import sys
import traceback
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from indextts.infer_v2_acc import IndexTTS2Acc as IndexTTS2
from indextts.infer_v2 import IndexTTS2


# --- 环境变量配置 ---
DEFAULT_SPK_AUDIO = os.environ.get("TTS_SPK_AUDIO", "examples/voice_template_02.wav")
MODEL_CFG = os.environ.get("INDEX_TTS_CFG", "checkpoints/config.yaml")
MODEL_DIR = os.environ.get("INDEX_TTS_MODEL_DIR", "checkpoints")
HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8000"))
DEVICE = os.environ.get("TTS_DEVICE", "cpu")

tts: IndexTTS2 | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    print(f">> Loading IndexTTS2 model (cfg={MODEL_CFG}, model_dir={MODEL_DIR})...")
    tts = IndexTTS2(
        cfg_path=MODEL_CFG, model_dir=MODEL_DIR, device=DEVICE, use_cuda_kernel=False,
        # use_torch_compile=True, use_gpt_cache=True, gpt_cache_size=128,
    )
    print(f">> Model loaded on device: {tts.device}")
    print(f">> Default speaker audio: {DEFAULT_SPK_AUDIO}")
    # tts.warm_up(DEFAULT_SPK_AUDIO)
    yield
    # shutdown
    print(">> Shutting down...")
    del tts
    tts = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="IndexTTS HTTP Service", version="1.0.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    try:
        result = tts.infer(
            spk_audio_prompt=DEFAULT_SPK_AUDIO,
            text=req.text.strip(),
            output_path=None,
            verbose=False
        )
        if result is None:
            raise HTTPException(status_code=500, detail="inference returned no result")

        sr, wav = result
        # wav shape: [samples, channels], torchaudio needs [channels, samples]
        wav_tensor = torch.from_numpy(wav.copy()).T.contiguous().to(torch.int16)

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_tensor, sr, format="wav")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=tts_output.wav"},
        )

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="inference failed")


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": tts.device if tts else "not loaded"}


if __name__ == "__main__":
    import uvicorn

    print(f">> Starting HTTP TTS service on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
