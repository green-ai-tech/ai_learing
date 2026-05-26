import request from './index'

export async function generateOutline(payload) {
  const response = await request.post('/outline/generate', payload)
  return response.data
}

export async function fetchOutlines(params = {}) {
  const response = await request.get('/outline', { params })
  return response.data
}

export async function fetchOutline(outlineId) {
  const response = await request.get(`/outline/${outlineId}`)
  return response.data
}

export async function generatePptx(outlineId) {
  const response = await request.post(`/outline/${outlineId}/generate-pptx`)
  return response.data
}

export async function downloadOutlineFile(outlineId, fileType) {
  const response = await request.get(`/outline/${outlineId}/download/${fileType}`, {
    responseType: 'blob',
  })
  const contentDisposition = response.headers['content-disposition'] || ''
  const match = contentDisposition.match(/filename="?([^"]+)"?/)
  const filename = match ? decodeURIComponent(match[1]) : `outline-${outlineId}.${fileType}`
  const url = URL.createObjectURL(response.data)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}
