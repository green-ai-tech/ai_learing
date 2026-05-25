<template>
  <main class="min-h-screen bg-gray-50">
    <div class="mx-auto max-w-[1400px] px-8 pt-2">
      <!-- 页面标题区 -->
      <div class="mb-2 w-full">
          <h1 class="text-2xl font-bold text-gray-900 text-center" >教学智能体应用精选</h1>
          <p class="mt-1 text-gray-300 text-center">为您的教学寻找AI解决方案，快速构建工作流。</p>
      </div>
      <!-- 分类过滤器 Tabs -->
      <div class="mb-2 flex items-center gap-2 overflow-x-auto pb-2 scrollbar-hide">
        <button
          v-for="cat in categories"
          :key="cat"
          class="whitespace-nowrap rounded-full px-4 py-1.5 text-sm font-medium transition-colors"
          :class="
            currentCategory === cat
              ? 'bg-gray-900 text-white'
              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-100'
          "
          @click="currentCategory = cat"
        >
          {{ cat }}
        </button>
      </div>

      <!-- 卡片网格布局 -->
      <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        <AppCard
          v-for="app in filteredApps"
          :key="app.id"
          :title="app.title"
          :description="app.description"
          :resource="app.resource"
          :tag="app.tag"
          :icon="app.icon"
          :icon-bg-class="app.iconBg"
          :tag-color-class="app.tagColor"
          :show-action="app.showAction"
        />
      </div>
    </div>
  </main>
</template>

<script setup>
import { ref, computed, onMounted, markRaw } from 'vue'
import {
  BarChart3,
  BookOpen,
  CalendarCheck,
  CircleHelp,
  ClipboardCheck,
  Eye,
  FileQuestion,
  FileText,
  FlaskConical,
  GraduationCap,
  Lightbulb,
  ListTree,
  NotebookPen,
  Presentation,
  Vote,
} from '@lucide/vue'
import AppCard from '../components/Card.vue'
import { fetchApps, fetchAppStages } from '../api/apps'

const iconMap = {
  BarChart3,
  BookOpen,
  CalendarCheck,
  CircleHelp,
  ClipboardCheck,
  Eye,
  FileQuestion,
  FileText,
  FlaskConical,
  GraduationCap,
  Lightbulb,
  ListTree,
  NotebookPen,
  Presentation,
  Vote,
}
const iconToneMap = {
  yellow: 'bg-yellow-500',
  orange: 'bg-orange-500',
  purple: 'bg-purple-500',
  blue: 'bg-blue-600',
  teal: 'bg-teal-500',
  red: 'bg-red-500',
  indigo: 'bg-indigo-500',
  cyan: 'bg-cyan-500',
  pink: 'bg-pink-500',
  emerald: 'bg-emerald-500',
}
const tagToneMap = {
  blue: 'bg-blue-50 text-blue-600',
  green: 'bg-green-50 text-green-600',
  violet: 'bg-violet-50 text-violet-600',
  slate: 'bg-slate-100 text-slate-600',
}
const legacyToneMap = {
  'bg-yellow-500': 'yellow',
  'bg-orange-500': 'orange',
  'bg-purple-500': 'purple',
  'bg-blue-600': 'blue',
  'bg-teal-500': 'teal',
  'bg-red-500': 'red',
  'bg-indigo-500': 'indigo',
  'bg-cyan-500': 'cyan',
  'bg-blue-50 text-blue-600': 'blue',
  'bg-green-50 text-green-600': 'green',
  'bg-violet-50 text-violet-600': 'violet',
  'bg-slate-100 text-slate-600': 'slate',
}

const currentCategory = ref('所有')
const categories = ref(['所有'])
const apps = ref([])

async function loadCategories() {
  try {
    categories.value = await fetchAppStages()
  } catch (error) {
    console.error('获取分类失败', error)
  }
}

async function loadApps() {
  try {
    const data = await fetchApps()
    apps.value = data.map((app) => ({
      ...app,
      icon: markRaw(iconMap[app.icon] || FileText),
      iconBg: iconToneMap[app.iconTone || legacyToneMap[app.iconBg]] || 'bg-blue-600',
      tagColor: tagToneMap[app.tagTone || legacyToneMap[app.tagColor]] || 'bg-gray-100 text-gray-600',
    }))
  } catch (error) {
    console.error('获取数据失败', error)
  }
}

onMounted(() => {
  loadCategories()
  loadApps()
})

const filteredApps = computed(() => {
  if (currentCategory.value === '所有') return apps.value
  return apps.value.filter((app) => app.stage === currentCategory.value)
})
</script>
