"""
Backend local do CifraShow: transcrição de áudio + estimativa de acordes.
Execute no computador Windows onde o CifraShow está rodando.

Observações:
- A transcrição usa faster-whisper (modelo baixado na primeira execução).
- Os acordes são estimativas por análise de cromagrama e podem precisar de revisão.
- Não transcreve automaticamente links protegidos/streaming; use um arquivo de áudio local.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel

APP_TITLE = "CifraShow Audio API"
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

app = FastAPI(title=APP_TITLE, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # API local; não exponha esta porta publicamente.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
    return _model


class Health(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model=MODEL_SIZE)


def transcrever(caminho: str) -> tuple[str, str, list[dict[str, Any]]]:
    """Transcreve o áudio e mantém os tempos das palavras para alinhar os acordes."""
    model = get_model()
    segments, info = model.transcribe(
        caminho,
        language="pt",
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
    )

    textos: list[str] = []
    trechos: list[dict[str, Any]] = []
    for segmento in segments:
        texto_segmento = (segmento.text or "").strip()
        if not texto_segmento:
            continue
        textos.append(texto_segmento)

        palavras: list[dict[str, Any]] = []
        for palavra in (segmento.words or []):
            texto_palavra = (palavra.word or "").strip()
            if not texto_palavra or palavra.start is None:
                continue
            palavras.append({
                "texto": texto_palavra,
                "inicio": round(float(palavra.start), 2),
                "fim": round(float(palavra.end if palavra.end is not None else palavra.start), 2),
            })

        trechos.append({
            "texto": texto_segmento,
            "inicio": round(float(segmento.start), 2),
            "fim": round(float(segmento.end), 2),
            "palavras": palavras,
        })

    return "\\n".join(textos), (info.language or "pt"), trechos


NOTAS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _perfil_acorde(raiz: int, menor: bool) -> np.ndarray:
    perfil = np.zeros(12, dtype=float)
    intervalos = (0, 3, 7) if menor else (0, 4, 7)
    for intervalo in intervalos:
        perfil[(raiz + intervalo) % 12] = 1.0
    return perfil


def estimar_acordes(caminho: str) -> tuple[list[dict[str, Any]], str]:
    """
    Estimativa básica: compara cromagrama com tríades maiores/menores.
    É uma aproximação, não substitui revisão musical, especialmente em samba,
    pagode, acordes com sétima, inversões e gravações com muitos instrumentos.
    """
    y, sr = librosa.load(caminho, sr=22050, mono=True)
    if len(y) < sr * 2:
        raise ValueError("O áudio precisa ter pelo menos 2 segundos.")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    templates = []
    labels = []
    for raiz, nota in enumerate(NOTAS):
        for menor in (False, True):
            templates.append(_perfil_acorde(raiz, menor))
            labels.append(nota + ("m" if menor else ""))
    templates_np = np.asarray(templates)
    norm_templates = templates_np / (np.linalg.norm(templates_np, axis=1, keepdims=True) + 1e-9)

    # Janelas de aproximadamente 1,5 s para reduzir trocas excessivas.
    frames_per_window = max(1, int(1.5 * sr / 2048))
    estimativas: list[dict[str, Any]] = []
    for inicio in range(0, chroma.shape[1], frames_per_window):
        trecho = chroma[:, inicio:inicio + frames_per_window]
        if trecho.size == 0:
            continue
        perfil = np.mean(trecho, axis=1)
        norma = np.linalg.norm(perfil)
        if norma <= 1e-9:
            continue
        perfil = perfil / norma
        scores = norm_templates @ perfil
        idx = int(np.argmax(scores))
        segundos = inicio * 2048 / sr
        acorde = labels[idx]
        if estimativas and estimativas[-1]["acorde"] == acorde:
            estimativas[-1]["fim"] = round((inicio + trecho.shape[1]) * 2048 / sr, 1)
            estimativas[-1]["confianca"] = round(
                max(estimativas[-1]["confianca"], float(scores[idx])), 2
            )
        else:
            estimativas.append({
                "inicio": round(segundos, 1),
                "fim": round((inicio + trecho.shape[1]) * 2048 / sr, 1),
                "acorde": acorde,
                "confianca": round(float(scores[idx]), 2),
            })

    # Evita devolver uma lista enorme em músicas longas.
    return estimativas[:250], "Estimativa por cromagrama; revise os acordes antes de usar."


@app.post("/analisar")
async def analisar_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    nome = (file.filename or "audio").lower()
    extensao = Path(nome).suffix
    permitidas = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm"}
    if extensao not in permitidas:
        raise HTTPException(status_code=400, detail="Formato de áudio não suportado.")

    caminho_temporario = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as tmp:
            caminho_temporario = tmp.name
            conteudo = await file.read()
            if not conteudo:
                raise HTTPException(status_code=400, detail="O arquivo está vazio.")
            if len(conteudo) > 100 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Arquivo maior que 100 MB.")
            tmp.write(conteudo)

        try:
            letra, idioma, trechos = transcrever(caminho_temporario)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na transcrição: {exc}",
            ) from exc

        try:
            acordes, aviso = estimar_acordes(caminho_temporario)
        except Exception as exc:
            acordes, aviso = [], f"Não foi possível estimar os acordes: {exc}"

        return {
            "letra": letra,
            "idioma": idioma,
            "acordes": acordes,
            "trechos": trechos,
            "cifra_sugerida": "  ".join(x["acorde"] for x in acordes),
            "aviso_acordes": aviso,
            "precisa_revisao": True,
        }
    finally:
        if caminho_temporario and os.path.exists(caminho_temporario):
            os.remove(caminho_temporario)
