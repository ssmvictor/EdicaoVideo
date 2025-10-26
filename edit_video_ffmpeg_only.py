#!/usr/bin/env python3
import argparse, subprocess, json, re, sys
from pathlib import Path

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def which(name):
    from shutil import which as _which
    return _which(name) is not None

def require_ffmpeg():
    if not which("ffmpeg") or not which("ffprobe"):
        print("ERROR: ffmpeg/ffprobe not found on PATH.", file=sys.stderr)
        sys.exit(2)

def extract_clean_wav_for_detection(input_video, out_wav, denoise=True, hp=80, lp=12000):
    af = []
    if denoise:
        af.append("afftdn=nr=10:nf=-25")
    if hp:
        af.append(f"highpass=f={hp}")
    if lp:
        af.append(f"lowpass=f={lp}")
    af_str = ",".join(af) if af else "anull"
    cmd = ["ffmpeg","-y","-i",str(input_video),"-vn","-af",af_str,"-ac","1","-ar","48000",str(out_wav)]
    code, out = run(cmd)
    if code != 0:
        print(out, file=sys.stderr)
        sys.exit(3)

def detect_silences_ffmpeg(wav_path, threshold_db=-38.0, min_len_s=1.0):
    cmd = ["ffmpeg","-hide_banner","-nostats","-i",str(wav_path),
           "-af", f"silencedetect=noise={threshold_db}dB:d={min_len_s}",
           "-f","null","-"]
    code, out = run(cmd)
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", out)]
    ends   = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", out)]
    silences = []
    i = j = 0
    while i < len(starts) and j < len(ends):
        if ends[j] <= starts[i]:
            j += 1
            continue
        silences.append((starts[i], ends[j]))
        i += 1
        j += 1
    return silences

def probe_duration_seconds(input_path):
    code, out = run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(input_path)])
    try:
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None

def build_keep_ranges_gentle(total_s, silences,
                             long_threshold_s=1.0,
                             reduce_ratio=0.6,
                             min_final_s=0.5,
                             max_final_s=None,
                             head_tail_ratio=0.5):
    keep = []
    cursor = 0.0
    for (s, e) in silences:
        if cursor < s:
            keep.append([cursor, s])
        L = max(0.0, e - s)
        if L < long_threshold_s:
            keep.append([s, e])
        else:
            final_len = max(L * reduce_ratio, min_final_s)
            if max_final_s is not None and max_final_s >= 0:
                final_len = min(final_len, max_final_s)
            head_keep = final_len * head_tail_ratio
            tail_keep = final_len - head_keep
            head_end = s + head_keep
            tail_start = e - tail_keep
            if head_end >= tail_start:
                mid = s + (L - final_len)/2.0
                keep.append([mid, mid + final_len])
            else:
                keep.append([s, head_end])
                keep.append([tail_start, e])
        cursor = e
    if cursor < total_s:
        keep.append([cursor, total_s])

    merged = []
    for seg in keep:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] <= prev[1] + 0.002:
                prev[1] = max(prev[1], seg[1])
            else:
                merged.append(seg)
    for seg in merged:
        seg[0] = max(0.0, seg[0])
        seg[1] = max(seg[0], seg[1])
    return merged

def build_filter_complex(keep_ranges, audio_chain, preset_audio_denoise=None):
    parts = []
    pair_labels = []
    for i,(s,e) in enumerate(keep_ranges):
        vlab = f"v{i}"
        alab = f"a{i}"
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[{vlab}]")
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[{alab}]")
        pair_labels.append(f"[{vlab}][{alab}]")  # interleaved v,a for concat
    n = len(keep_ranges)
    if n == 0:
        raise RuntimeError("No ranges to keep.")
    # concat expects interleaved inputs: [v0][a0][v1][a1]... when v=1:a=1
    parts.append("".join(pair_labels) + f"concat=n={n}:v=1:a=1[vcat][acat]")

    # optional audio denoise/EQ pre-dynamics
    denoise = (preset_audio_denoise + ",") if preset_audio_denoise else ""
    parts.append(f"[acat]{denoise}{audio_chain}[aout]")
    return ";".join(parts)

def main():
    ap = argparse.ArgumentParser(description="Edit video with ffmpeg only: gently shorten long pauses and normalize audio.")
    ap.add_argument("-i","--input", required=True)
    ap.add_argument("-o","--output", required=True)
    ap.add_argument("--silence-threshold", type=float, default=-38.0)
    ap.add_argument("--min-silence", type=int, default=1000)
    ap.add_argument("--long-threshold", type=int, default=1000)
    ap.add_argument("--reduce-ratio", type=float, default=0.6)
    ap.add_argument("--min-final-silence", type=int, default=500)
    ap.add_argument("--max-final-silence", type=int, default=1400)
    ap.add_argument("--head-tail-ratio", type=float, default=0.5)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    require_ffmpeg()
    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 1) Detect on cleaned WAV
    wav = outp.with_suffix(".detect.wav")
    extract_clean_wav_for_detection(inp, wav, denoise=True)

    # 2) Silences
    silences = detect_silences_ffmpeg(wav, threshold_db=args.silence_threshold, min_len_s=args.min_silence/1000.0)
    dur = probe_duration_seconds(inp) or 0.0

    # 3) Keep ranges
    max_final = None if args.max_final_silence is not None and args.max_final_silence < 0 else args.max_final_silence/1000.0
    keep = build_keep_ranges_gentle(
        total_s=dur,
        silences=silences,
        long_threshold_s=args.long_threshold/1000.0,
        reduce_ratio=args.reduce_ratio,
        min_final_s=args.min_final_silence/1000.0,
        max_final_s=max_final,
        head_tail_ratio=args.head_tail_ratio
    )

    # 4) Filter graph
    audio_chain = "dynaudnorm=f=301:g=6,acompressor=threshold=-18dB:ratio=2.5:attack=5:release=120:makeup=2.5,loudnorm=I=-16:TP=-1.5:LRA=11"
    # set f=301 (odd) to avoid 'filter size' warning on some builds
    post_denoise = "afftdn=nr=8:nf=-25,highpass=f=80,lowpass=f=12000"
    fc = build_filter_complex(keep, audio_chain=audio_chain, preset_audio_denoise=post_denoise)

    cmd = [
        "ffmpeg","-y","-i",str(inp),
        "-filter_complex", fc,
        "-map","[vcat]","-map","[aout]",
        "-c:v","libx264","-preset",args.preset,"-crf",str(args.crf),
        "-c:a","aac","-b:a","192k",
        "-movflags","+faststart",
        str(outp)
    ]
    code, out = run(cmd)
    if code != 0:
        print(out, file=sys.stderr)
        sys.exit(code)

    try:
        wav.unlink()
    except Exception:
        pass

    print("✅ Done.")
    print(f"➡️ Output saved to: {outp}")

if __name__ == "__main__":
    main()
