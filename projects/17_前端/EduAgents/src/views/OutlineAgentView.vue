<template>
  <main class="min-h-screen bg-gray-50 pb-16">
    <div class="mx-auto max-w-[1400px] px-6 py-6">
      <div class="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">课程大纲生成智能体</h1>
          <p class="mt-1 text-sm text-gray-500">结构化生成教学大纲，并导出为 XLSX 或 PPTX。</p>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <button class="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100" @click="loadHistory">
            <History :size="16" />
            查看历史
          </button>
          <button class="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100" @click="backToEdit">
            <ArrowLeft :size="16" />
            返回编辑
          </button>
        </div>
      </div>

      <div v-if="errorMessage" class="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        {{ errorMessage }}
      </div>

      <div class="grid gap-5 lg:grid-cols-[360px_minmax(0,1fr)_300px]">
        <section class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
          <div class="mb-4 flex items-center justify-between">
            <h2 class="text-base font-semibold text-gray-900">课程信息</h2>
            <span v-if="historyMode" class="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-600">历史模式</span>
          </div>

          <div class="space-y-4">
            <label class="block">
              <span class="mb-1 block text-sm font-medium text-gray-700">课程主题</span>
              <input v-model="form.course_title" :readonly="historyMode" class="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
            </label>

            <label class="block">
              <span class="mb-1 block text-sm font-medium text-gray-700">课程介绍</span>
              <textarea v-model="form.course_description" :readonly="historyMode" rows="4" class="w-full resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
            </label>

            <div class="grid grid-cols-2 gap-3">
              <label class="block">
                <span class="mb-1 block text-sm font-medium text-gray-700">学段</span>
                <input v-model="form.stage" :readonly="historyMode" class="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
              </label>
              <label class="block">
                <span class="mb-1 block text-sm font-medium text-gray-700">课时</span>
                <input v-model.number="form.total_hours" :readonly="historyMode" type="number" min="1" class="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
              </label>
            </div>

            <div class="grid grid-cols-2 gap-3">
              <label class="block">
                <span class="mb-1 block text-sm font-medium text-gray-700">目标学生</span>
                <input v-model="form.target_students" :readonly="historyMode" class="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
              </label>
              <label class="block">
                <span class="mb-1 block text-sm font-medium text-gray-700">难度</span>
                <select v-model="form.difficulty" :disabled="historyMode" class="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500">
                  <option>基础</option>
                  <option>中等</option>
                  <option>进阶</option>
                </select>
              </label>
            </div>

            <label class="block">
              <span class="mb-1 block text-sm font-medium text-gray-700">教学目标</span>
              <textarea v-model="form.teaching_goals" :readonly="historyMode" rows="3" class="w-full resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
            </label>

            <label class="block">
              <span class="mb-1 block text-sm font-medium text-gray-700">知识点</span>
              <textarea v-model="form.key_points" :readonly="historyMode" rows="3" class="w-full resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
            </label>

            <label class="block">
              <span class="mb-1 block text-sm font-medium text-gray-700">难点</span>
              <textarea v-model="form.difficult_points" :readonly="historyMode" rows="3" class="w-full resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-500" />
            </label>
          </div>

          <button
            class="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
            :disabled="generating || historyMode"
            @click="submitOutline"
          >
            <LoaderCircle v-if="generating" :size="16" class="animate-spin" />
            <Play v-else :size="16" />
            生成大纲
          </button>
        </section>

        <section class="min-h-[720px] rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
          <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 class="text-base font-semibold text-gray-900">结构化大纲</h2>
              <p v-if="currentRecord" class="mt-1 text-xs text-gray-400">记录 #{{ currentRecord.id }} · {{ currentRecord.conversation_id }}</p>
            </div>
            <div class="flex flex-wrap items-center gap-2">
              <button class="inline-flex items-center gap-2 rounded-lg border border-gray-200 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300" :disabled="!currentRecord" @click="downloadXlsx">
                <FileSpreadsheet :size="16" />
                下载 XLSX
              </button>
              <button class="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300" :disabled="!currentRecord || pptxGenerating" @click="createPptx">
                <LoaderCircle v-if="pptxGenerating" :size="16" class="animate-spin" />
                <Presentation v-else :size="16" />
                生成 PPTX
              </button>
              <button class="inline-flex items-center gap-2 rounded-lg border border-gray-200 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-300" :disabled="!currentRecord?.pptx_file_path" @click="downloadPptx">
                <Download :size="16" />
                下载 PPTX
              </button>
            </div>
          </div>

          <div v-if="generating" class="flex h-[560px] items-center justify-center rounded-lg border border-dashed border-gray-200 text-sm text-gray-500">
            <LoaderCircle :size="18" class="mr-2 animate-spin" />
            正在生成结构化教学大纲...
          </div>

          <div v-else-if="!outline" class="flex h-[560px] items-center justify-center rounded-lg border border-dashed border-gray-200 text-sm text-gray-400">
            等待生成结果
          </div>

          <div v-else class="space-y-5">
            <div class="rounded-lg bg-gray-50 p-4">
              <h3 class="text-xl font-bold text-gray-900">{{ outline.course_title }}</h3>
              <p class="mt-2 text-sm leading-6 text-gray-600">{{ outline.course_description }}</p>
              <div class="mt-3 flex flex-wrap gap-2 text-xs text-gray-600">
                <span class="rounded-md bg-white px-2 py-1">对象：{{ outline.target_students }}</span>
                <span class="rounded-md bg-white px-2 py-1">课时：{{ outline.total_hours }}</span>
                <span class="rounded-md bg-white px-2 py-1">难度：{{ outline.difficulty }}</span>
              </div>
            </div>

            <div class="grid gap-4 md:grid-cols-2">
              <InfoBlock title="教学目标" :items="outline.teaching_goals" />
              <InfoBlock title="教学重点" :items="outline.key_points" />
              <InfoBlock title="教学难点" :items="outline.difficult_points" />
              <InfoBlock title="考核方式" :items="outline.assessment_methods" />
            </div>

            <div>
              <h3 class="mb-3 text-base font-semibold text-gray-900">章节目录</h3>
              <div class="space-y-3">
                <div v-for="chapter in outline.chapters" :key="chapter.title" class="rounded-lg border border-gray-200 p-4">
                  <div class="flex items-start justify-between gap-3">
                    <div>
                      <h4 class="font-semibold text-gray-900">{{ chapter.title }}</h4>
                      <p class="mt-1 text-sm leading-6 text-gray-500">{{ chapter.description }}</p>
                    </div>
                    <span class="shrink-0 rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-600">{{ chapter.hours }}课时</span>
                  </div>
                  <div class="mt-3 grid gap-2 md:grid-cols-2">
                    <div v-for="section in chapter.sections" :key="section.title" class="rounded-md bg-gray-50 p-3">
                      <div class="text-sm font-medium text-gray-800">{{ section.title }}</div>
                      <div class="mt-1 text-xs leading-5 text-gray-500">{{ section.description }}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="grid gap-4 md:grid-cols-2">
              <InfoBlock title="教学要求" :items="outline.teaching_requirements" />
              <InfoBlock title="教学方法" :items="outline.teaching_methods" />
            </div>
          </div>
        </section>

        <aside class="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <div class="mb-3 flex items-center justify-between">
            <h2 class="text-base font-semibold text-gray-900">历史记录</h2>
            <button class="rounded-md p-1.5 text-gray-500 hover:bg-gray-100" @click="loadHistory">
              <RefreshCw :size="16" />
            </button>
          </div>
          <div v-if="historyLoading" class="py-8 text-center text-sm text-gray-400">加载中...</div>
          <div v-else-if="histories.length === 0" class="py-8 text-center text-sm text-gray-400">暂无记录</div>
          <div v-else class="space-y-2">
            <button
              v-for="item in histories"
              :key="item.id"
              class="block w-full rounded-lg border border-gray-200 px-3 py-3 text-left hover:border-indigo-200 hover:bg-indigo-50"
              :class="currentRecord?.id === item.id ? 'border-indigo-300 bg-indigo-50' : 'bg-white'"
              @click="selectHistory(item.id)"
            >
              <div class="truncate text-sm font-medium text-gray-900">{{ item.outline_json?.course_title || item.user_input?.course_title }}</div>
              <div class="mt-1 text-xs text-gray-500">{{ item.status }} · {{ formatTime(item.created_at) }}</div>
            </button>
          </div>
        </aside>
      </div>
    </div>
  </main>
</template>

<script setup>
import { computed, defineComponent, h, onMounted, reactive, ref } from 'vue'
import {
  ArrowLeft,
  Download,
  FileSpreadsheet,
  History,
  LoaderCircle,
  Play,
  Presentation,
  RefreshCw,
} from '@lucide/vue'
import {
  downloadOutlineFile,
  fetchOutline,
  fetchOutlines,
  generateOutline,
  generatePptx,
} from '../api/outline'

const InfoBlock = defineComponent({
  props: {
    title: { type: String, required: true },
    items: { type: Array, default: () => [] },
  },
  setup(props) {
    return () =>
      h('div', { class: 'rounded-lg border border-gray-200 p-4' }, [
        h('h3', { class: 'mb-2 text-sm font-semibold text-gray-900' }, props.title),
        h(
          'ul',
          { class: 'space-y-2 text-sm leading-6 text-gray-600' },
          props.items?.length
            ? props.items.map((item) => h('li', { class: 'rounded-md bg-gray-50 px-3 py-2' }, item))
            : [h('li', { class: 'text-gray-400' }, '暂无')],
        ),
      ])
  },
})

const defaultForm = {
  course_title: '人工智能基础',
  course_description: '面向学生系统介绍人工智能的发展、核心概念、典型算法和实践应用。',
  stage: '高中 / 大学通识',
  total_hours: 32,
  target_students: '具备基础信息技术素养的学生',
  difficulty: '中等',
  teaching_goals: '理解人工智能基本概念\n掌握典型算法思想\n能够完成简单 AI 应用分析',
  key_points: '人工智能概述\n机器学习基础\n神经网络入门\n生成式 AI 应用',
  difficult_points: '算法思想抽象\n模型训练与评估\n真实场景迁移应用',
}

const form = reactive({ ...defaultForm })
const currentRecord = ref(null)
const histories = ref([])
const generating = ref(false)
const pptxGenerating = ref(false)
const historyLoading = ref(false)
const historyMode = ref(false)
const errorMessage = ref('')

const outline = computed(() => currentRecord.value?.outline_json || null)

async function submitOutline() {
  errorMessage.value = ''
  generating.value = true
  try {
    const data = await generateOutline({ ...form })
    currentRecord.value = data
    historyMode.value = false
    await loadHistory()
  } catch (error) {
    errorMessage.value = error.response?.data?.detail || '大纲生成失败，请检查后端服务、MySQL 和模型配置。'
  } finally {
    generating.value = false
  }
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const data = await fetchOutlines({ limit: 50 })
    histories.value = data.items || []
  } catch (error) {
    errorMessage.value = error.response?.data?.detail || '历史记录加载失败。'
  } finally {
    historyLoading.value = false
  }
}

async function selectHistory(id) {
  errorMessage.value = ''
  try {
    const data = await fetchOutline(id)
    currentRecord.value = data
    Object.assign(form, toEditableForm(data.user_input || {}))
    historyMode.value = true
  } catch (error) {
    errorMessage.value = error.response?.data?.detail || '历史记录读取失败。'
  }
}

function backToEdit() {
  historyMode.value = false
}

async function downloadXlsx() {
  if (!currentRecord.value) return
  try {
    await downloadOutlineFile(currentRecord.value.id, 'xlsx')
    await refreshCurrent()
  } catch (error) {
    errorMessage.value = error.response?.data?.detail || 'XLSX 下载失败。'
  }
}

async function createPptx() {
  if (!currentRecord.value) return
  pptxGenerating.value = true
  errorMessage.value = ''
  try {
    currentRecord.value = await generatePptx(currentRecord.value.id)
    await loadHistory()
  } catch (error) {
    errorMessage.value = error.response?.data?.detail || 'PPTX 生成失败。'
  } finally {
    pptxGenerating.value = false
  }
}

async function downloadPptx() {
  if (!currentRecord.value?.pptx_file_path) return
  try {
    await downloadOutlineFile(currentRecord.value.id, 'pptx')
  } catch (error) {
    errorMessage.value = error.response?.data?.detail || 'PPTX 下载失败。'
  }
}

async function refreshCurrent() {
  if (!currentRecord.value) return
  currentRecord.value = await fetchOutline(currentRecord.value.id)
}

function toEditableForm(input) {
  return {
    ...defaultForm,
    ...input,
    teaching_goals: toText(input.teaching_goals),
    key_points: toText(input.key_points),
    difficult_points: toText(input.difficult_points),
  }
}

function toText(value) {
  if (Array.isArray(value)) return value.join('\n')
  return value || ''
}

function formatTime(value) {
  if (!value) return ''
  return new Date(value).toLocaleString()
}

onMounted(loadHistory)
</script>
