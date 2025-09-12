# ComfyUI-Ollama-Enhancer 🚀

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes powered by [Ollama](https://ollama.ai)  
to automatically **generate and refine positive/negative prompts** for diffusion models like **Stable Diffusion XL** and **Flux**.

---

## ✨ Features

- 🔹 **Positive + Negative prompt generation** with Ollama  
- 🔹 Works with **Stable Diffusion XL** and **Flux** checkpoints  
- 🔹 Supports **Ollama local / remote endpoints** (e.g. `http://localhost:11434` or your server)  
- 🔹 JSON-based structured output for easy workflow integration  
- 🔹 Optional **CPU forcing** to reduce VRAM usage  
- 🔹 Plug-and-play with existing ComfyUI workflows  

---

## 📦 Installation

Clone this repo inside your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/ComfyUI-Ollama-Enhancer.git