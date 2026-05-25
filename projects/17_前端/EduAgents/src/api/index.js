import axios from 'axios'

const request = axios.create({
  baseURL: '/api',
  timeout: 10000,
})

request.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API 请求失败:', error.message)
    return Promise.reject(error)
  }
)

export default request
