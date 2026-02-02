<template>
  <div class="ocr-uploader">
    <header class="header">
      <h1>DeepSeek OCR 2.0</h1>
      <p class="subtitle">PDF文档智能识别系统</p>
    </header>

    <div class="server-config">
      <label>服务器地址：</label>
      <input 
        v-model="serverUrl" 
        type="text" 
        class="server-input"
      >
      <button @click="checkHealth" :disabled="checkingHealth" class="health-btn">
        {{ checkingHealth ? '检测中...' : '检测连接' }}
      </button>
      <span :class="['status-indicator', healthStatus]"></span>
    </div>

    <div class="upload-area" 
         @dragover.prevent="isDragging = true"
         @dragleave.prevent="isDragging = false"
         @drop.prevent="handleDrop"
         :class="{ 'dragging': isDragging }">
      <input 
        ref="fileInput"
        type="file" 
        multiple 
        accept=".pdf"
        @change="handleFileSelect"
        id="file-input"
        class="file-input"
      >
      <label for="file-input" class="upload-label">
        <div class="upload-icon">📄</div>
        <p class="upload-text">拖拽PDF文件到此处，或 <span>点击选择文件</span></p>
        <p class="upload-hint">支持单个或多个PDF文件</p>
      </label>
    </div>

    <div v-if="selectedFiles.length > 0" class="file-list">
      <h3>已选择文件 ({{ selectedFiles.length }})</h3>
      <ul>
        <li v-for="(file, index) in selectedFiles" :key="index" class="file-item">
          <span class="file-icon">📄</span>
          <span class="file-name">{{ file.name }}</span>
          <span class="file-size">{{ formatFileSize(file.size) }}</span>
          <button @click="removeFile(index)" class="remove-btn">×</button>
        </li>
      </ul>
      <div class="action-buttons">
        <button @click="clearFiles" class="btn btn-secondary">清空</button>
        <button @click="uploadFiles" :disabled="uploading || selectedFiles.length === 0" class="btn btn-primary">
          {{ uploading ? `处理中 ${uploadProgress}%` : '开始OCR识别' }}
        </button>
      </div>
    </div>

    <div v-if="taskId" class="task-status">
      <h3>处理状态</h3>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
      </div>
      <p class="status-message">{{ statusMessage }}</p>
      
      <div v-if="taskResults.length > 0" class="results-section">
        <h4>处理结果</h4>
        <ul>
          <li v-for="(result, index) in taskResults" :key="index" class="result-item">
            <span :class="['result-icon', result.status]">{{ result.status === 'completed' ? '✓' : '✗' }}</span>
            <span class="result-name">{{ result.filename }}</span>
            <span :class="['result-status', result.status]">
              {{ result.status === 'completed' ? '成功' : '失败' }}
            </span>
          </li>
        </ul>
        <button @click="downloadResults" class="btn btn-download">
          下载全部结果
        </button>
      </div>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import axios from 'axios'
import { config } from '../config'

const serverUrl = ref(config.serverUrl)
const fileInput = ref(null)
const selectedFiles = ref([])
const isDragging = ref(false)
const uploading = ref(false)
const uploadProgress = ref(0)
const taskId = ref('')
const progress = ref(0)
const statusMessage = ref('')
const error = ref('')
const checkingHealth = ref(false)
const healthStatus = ref('unknown')
const taskResults = ref([])
const pollInterval = ref(null)

const checkHealth = async () => {
  checkingHealth.value = true
  healthStatus.value = 'unknown'
  error.value = ''
  
  try {
    const response = await axios.get(`${serverUrl.value}/api/health`)
    if (response.data.status === 'ok') {
      healthStatus.value = 'connected'
    } else {
      healthStatus.value = 'error'
      error.value = response.data.message || '模型加载失败'
    }
  } catch (err) {
    healthStatus.value = 'error'
    error.value = `无法连接到服务器: ${err.message}`
  } finally {
    checkingHealth.value = false
  }
}

const handleFileSelect = (event) => {
  const files = Array.from(event.target.files)
  addFiles(files)
  event.target.value = ''
}

const handleDrop = (event) => {
  isDragging.value = false
  const files = Array.from(event.dataTransfer.files).filter(f => f.type === 'application/pdf')
  if (files.length === 0) {
    error.value = '只支持PDF文件'
    return
  }
  addFiles(files)
}

const addFiles = (files) => {
  const pdfFiles = files.filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))
  selectedFiles.value = [...selectedFiles.value, ...pdfFiles]
  error.value = ''
}

const removeFile = (index) => {
  selectedFiles.value.splice(index, 1)
}

const clearFiles = () => {
  selectedFiles.value = []
  taskId.value = ''
  progress.value = 0
  statusMessage.value = ''
  taskResults.value = []
  error.value = ''
  if (pollInterval.value) {
    clearInterval(pollInterval.value)
    pollInterval.value = null
  }
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const uploadFiles = async () => {
  if (selectedFiles.value.length === 0) return
  
  uploading.value = true
  uploadProgress.value = 0
  error.value = ''
  taskResults.value = []
  
  const formData = new FormData()
  selectedFiles.value.forEach(file => {
    formData.append('files', file)
  })
  
  try {
    const response = await axios.post(`${serverUrl.value}/api/ocr`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    
    taskId.value = response.data.task_id
    statusMessage.value = '已提交任务，等待处理...'
    
    startPolling()
    
  } catch (err) {
    error.value = `上传失败: ${err.response?.data?.error || err.message}`
    uploading.value = false
  }
}

const startPolling = () => {
  pollInterval.value = setInterval(async () => {
    try {
      const response = await axios.get(`${serverUrl.value}/api/status/${taskId.value}`)
      const status = response.data
      
      progress.value = status.progress || 0
      statusMessage.value = status.message || '处理中...'
      
      if (status.status === 'completed') {
        clearInterval(pollInterval.value)
        pollInterval.value = null
        uploading.value = false
        taskResults.value = status.results || []
      } else if (status.status === 'failed') {
        clearInterval(pollInterval.value)
        pollInterval.value = null
        uploading.value = false
        error.value = status.message || '处理失败'
      }
      
    } catch (err) {
      clearInterval(pollInterval.value)
      pollInterval.value = null
      uploading.value = false
      error.value = `获取状态失败: ${err.message}`
    }
  }, 2000)
}

const downloadResults = async () => {
  if (!taskId.value) return
  
  try {
    const response = await axios.get(`${serverUrl.value}/api/download/${taskId.value}`, {
      responseType: 'blob'
    })
    
    const blob = new Blob([response.data], { type: 'application/zip' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `ocr_results_${taskId.value}.zip`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    
  } catch (err) {
    error.value = `下载失败: ${err.message}`
  }
}
</script>

<style scoped>
.ocr-uploader {
  background: white;
  border-radius: 20px;
  padding: 50px;
  box-shadow: 0 25px 80px rgba(0, 0, 0, 0.15);
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 2.5em;
  background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 10px;
}

.subtitle {
  color: #666;
  font-size: 1.1em;
}

.server-config {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 25px;
  padding: 20px;
  background: #f8fafc;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}

.server-config label {
  font-weight: 600;
  color: #475569;
  white-space: nowrap;
}

.server-input {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.3s ease;
}

.server-input:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
}

.health-btn {
  padding: 8px 16px;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.health-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.status-indicator.unknown {
  background: #ccc;
}

.status-indicator.connected {
  background: #4caf50;
  box-shadow: 0 0 8px #4caf50;
}

.status-indicator.error {
  background: #f44336;
  box-shadow: 0 0 8px #f44336;
}

.upload-area {
  border: 3px dashed #cbd5e1;
  border-radius: 16px;
  padding: 70px 50px;
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
  margin-bottom: 25px;
  background: #fafbfc;
}

.upload-area:hover,
.upload-area.dragging {
  border-color: #3498db;
  background: rgba(52, 152, 219, 0.05);
}

.file-input {
  display: none;
}

.upload-label {
  cursor: pointer;
}

.upload-icon {
  font-size: 80px;
  margin-bottom: 25px;
  transition: transform 0.3s ease;
}

.upload-area:hover .upload-icon {
  transform: scale(1.1);
}

.upload-text {
  font-size: 18px;
  color: #333;
  margin-bottom: 10px;
}

.upload-text span {
  color: #3498db;
  font-weight: 600;
}

.upload-hint {
  color: #999;
  font-size: 14px;
}

.file-list {
  margin-bottom: 20px;
}

.file-list h3 {
  margin-bottom: 15px;
  color: #333;
}

.file-item {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  background: #f8fafc;
  border-radius: 12px;
  margin-bottom: 12px;
  border: 1px solid #e2e8f0;
  transition: all 0.2s ease;
}

.file-item:hover {
  border-color: #3498db;
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.1);
}

.file-icon {
  margin-right: 10px;
  font-size: 20px;
}

.file-name {
  flex: 1;
  color: #333;
}

.file-size {
  color: #999;
  margin-right: 15px;
}

.remove-btn {
  width: 28px;
  height: 28px;
  border: none;
  background: #ff6b6b;
  color: white;
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
}

.action-buttons {
  display: flex;
  gap: 15px;
  margin-top: 25px;
  justify-content: flex-end;
}

@media (max-width: 640px) {
  .action-buttons {
    flex-direction: column;
  }
  
  .btn {
    width: 100%;
    text-align: center;
  }
}

.btn {
  padding: 14px 35px;
  border: none;
  border-radius: 10px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
}

.btn-secondary {
  background: #f1f5f9;
  color: #475569;
  border: 2px solid #e2e8f0;
}

.btn-secondary:hover {
  background: #e2e8f0;
}

.btn-download {
  background: #10b981;
  color: white;
  margin-top: 20px;
}

.btn-download:hover {
  background: #059669;
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
}

.task-status {
  margin-top: 30px;
  padding: 25px;
  background: #f8fafc;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
}

.task-status h3 {
  margin-bottom: 15px;
  color: #333;
}

.progress-bar {
  height: 24px;
  background: #e2e8f0;
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 15px;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.progress-fill {
  height: 100%;
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  transition: width 0.3s ease;
}

.status-message {
  color: #666;
  text-align: center;
}

.results-section {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.results-section h4 {
  margin-bottom: 15px;
  color: #333;
}

.result-item {
  display: flex;
  align-items: center;
  padding: 10px 0;
}

.result-icon {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 10px;
  font-size: 14px;
  color: white;
}

.result-icon.completed {
  background: #4caf50;
}

.result-icon.failed {
  background: #f44336;
}

.result-name {
  flex: 1;
  color: #333;
}

.result-status {
  font-weight: 500;
}

.result-status.completed {
  color: #4caf50;
}

.result-status.failed {
  color: #f44336;
}

.error-message {
  margin-top: 25px;
  padding: 18px 25px;
  background: #fef2f2;
  color: #dc2626;
  border-radius: 12px;
  text-align: center;
  border: 1px solid #fecaca;
  font-weight: 500;
}
</style>
