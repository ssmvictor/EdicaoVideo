# EdicaoVideo
Python para editar o vídeo para youtube


python edit_video_ffmpeg_only.py -i "IAStudio.mp4" -o "IAStudio_edit.mp4" `
  --silence-threshold -38 --min-silence 1100 --long-threshold 1100 `
  --reduce-ratio 0.70 --min-final-silence 600 --max-final-silence -1 `
  --head-tail-ratio 0.5 --preset medium --crf 17

