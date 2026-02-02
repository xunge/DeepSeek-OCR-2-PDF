# DeepSeek OCR 2.0

PDF 文档智能识别系统

## 项目结构

```
DeepSeek-OCR2-vllm/
├── config.py              # 后端配置文件
├── ocr_api.py             # Flask 后端服务
├── deepencoderv2/         # 编码器模块
├── process/               # 图像处理模块
├── frontend/              # Vue 前端
│   ├── src/
│   │   ├── config.js      # 前端配置文件（修改服务器地址）
│   │   ├── App.vue
│   │   └── components/
│   │       └── OcrUploader.vue
│   └── package.json
└── uploads/               # 上传文件目录
└── outputs/               # 输出文件目录
```

## 配置

```Shell
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2
```

- download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) 
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

### 后端配置 (config.py)

```python
FLASK_HOST = "0.0.0.0"     # 监听地址
FLASK_PORT = 5000          # 端口
MODEL_PATH = "..."         # 模型路径
PROMPT = "..."             # OCR 提示词
```

### 前端配置 (frontend/src/config.js)

```javascript
export const config = {
  serverUrl: 'http://jiang3090_4.xungejiang.com:5000'  // 后端服务器地址
}
```

**注意**: 修改前端配置后需重启开发服务器。

## 启动

### 1. 启动后端服务

```bash
python ocr_api.py
```

后端将在 `http://0.0.0.0:5000` 启动。

### 2. 启动前端开发服务器

```bash
cd frontend
npm install
npm run dev
```

前端将在 `http://localhost:3000` 启动。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/ocr` | POST | 上传 PDF 文件进行 OCR 识别 |
| `/api/status/<task_id>` | GET | 查询任务状态 |
| `/api/download/<task_id>` | GET | 下载结果压缩包 |

## 使用方法

1. 修改 `frontend/src/config.js` 中的服务器地址
2. 启动后端服务 `python ocr_api.py`
3. 启动前端 `cd frontend && npm run dev`
4. 打开浏览器访问前端地址
5. 输入服务器地址，点击"检测连接"
6. 拖拽或选择 PDF 文件，点击"开始 OCR 识别"
