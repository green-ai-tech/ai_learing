import request from './index'

export async function fetchApps(category) {
  const response = await request.get('/apps', {
    params: { category },
  })
  return response.data
}

export async function fetchAppStages() {
  const response = await request.get('/app-stages')
  return response.data
}
