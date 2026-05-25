<template>
  <div
    class="group relative flex min-h-[220px] flex-col justify-between rounded-lg border border-gray-200 bg-white p-5 shadow-sm transition-all hover:-translate-y-1 hover:shadow-md"
  >
    <!-- 顶部区域 -->
    <div class="mb-4">
      <!-- 图标与标题行 -->
      <div class="mb-3 flex items-start justify-between">
        <div
          class="flex h-10 w-10 items-center justify-center rounded-lg text-white shadow-sm"
          :class="iconBgClass"
        >
          <component :is="icon" :size="20" />
        </div>
        <span
          class="rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
          :class="tagColorClass"
        >
          {{ tag }}
        </span>
      </div>

      <!-- 文本内容 -->
      <h3 class="mb-2 text-base font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">
        {{ title }}
      </h3>
      <p class="line-clamp-3 text-sm leading-relaxed text-gray-500">
        {{ description }}
      </p>
      <div class="mt-4 rounded-md bg-gray-50 px-3 py-2 text-xs leading-relaxed text-gray-500">
        <span class="font-medium text-gray-700">关联资源</span>
        <span class="ml-1">{{ resource }}</span>
      </div>
    </div>

    <!-- 底部操作区 (可选，参考截图中的“使用模板”按钮) -->
    <div v-if="showAction" class="mt-auto pt-4">
      <button class="w-full rounded-lg bg-indigo-600 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700">
        + 使用模板
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  description: { type: String, required: true },
  resource: { type: String, default: '无' },
  tag: { type: String, default: 'Workflow' },
  icon: { type: [Object, Function], required: true },
  iconBgClass: { type: String, default: 'bg-blue-500' },
  tagColorClass: { type: String, default: 'bg-gray-100 text-gray-600' },
  showAction: { type: Boolean, default: false },
})
</script>
