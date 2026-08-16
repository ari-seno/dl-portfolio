# Project Memory (dl-portfolio)

## Lingkungan & Hardware (PENTING)

- **Selalu cek `which python` dari DALAM direktori project** sebelum menganalisis hardware atau menjalankan training. Ada 2 env Python:
  - Global pyenv (`python` default): torch `2.13.0+cu130`, **tidak ada XPU**
  - `dl-portfolio` (`~/.pyenv/versions/3.11.0/envs/dl-portfolio`): torch `2.13.0+xpu`, **Intel Arc GPU ~13.7GB VRAM**
- Project yang punya `.python-version` (mis. `08-indobert-finetuning`) otomatis memakai env tersebut saat `cd` ke dalamnya.
- Gunakan interpreter eksplisit untuk keamanan:
  `PY=~/.pyenv/versions/3.11.0/envs/dl-portfolio/bin/python`
- Pelajaran: menganalisis hardware dari root repo (tanpa `.python-version`) menghasilkan kesimpulan yang salah (menyangka CPU-only padahal ada Intel Arc via XPU). Tanda-tanda yang harus diperhatikan: `/dev/dri/renderD128` (GPU Intel), kode yang sudah memprioritaskan XPU, dan torch versi `+xpu`/`+cu130` yang berbeda per env.

## Konvensi

- `get_device()` di project 08: prioritas XPU > CUDA > CPU (sudah benar di `src/train.py`).
- Gunakan interpreter `dl-portfolio` untuk semua training PyTorch di project ini.