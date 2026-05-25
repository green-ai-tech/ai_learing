import request from './index'

export async function fetchApps(stageFilter) {
  const response = await request.post('/apps', {}, { params: { stage_filter: stageFilter } })
  return response.data
}

export async function fetchAppStages() {
  const response = await request.get('/app-stages')
  return response.data
}
