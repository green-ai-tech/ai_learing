<template>
  <header class="sticky top-0 z-50 w-full border-b border-gray-200 bg-white/80 backdrop-blur-md">
    <div class="mx-auto flex h-16 max-w-[1400px] items-center justify-between px-6">
      <!-- 左侧 Logo -->
      <div class="flex items-center gap-2 font-bold text-xl tracking-tight text-gray-900">
        <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
          <Sparkles :size="18" />
        </div>
        <span>智慧教学平台</span>
      </div>

      <!-- 中间导航 (居中) -->
      <nav class="hidden md:flex items-center gap-1 rounded-full bg-gray-100 p-1">
        <router-link
          v-for="(item, index) in navItems"
          :key="index"
          :to="item.to"
          class="flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all"
          :class="
            activeTab === item.label
              ? 'bg-white text-blue-500 shadow-sm'
              : 'text-gray-500 hover:text-gray-900 hover:bg-gray-200/50'
          "
          @click="activeTab = item.label"
        >
          <component :is="item.icon" :size="16" />
          {{ item.label }}
        </router-link>
      </nav>

      <!-- 右侧用户区 -->
      <div class="flex items-center gap-4">
        <button class="rounded-full p-2 text-gray-500 hover:bg-gray-100">
          <Bell :size="20" />
        </button>
        <div class="h-8 w-8 overflow-hidden rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 ring-2 ring-white">
          <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix" alt="User" />
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref } from 'vue';
import { Sparkles, Bell, LayoutGrid, BookOpen, Hammer } from '@lucide/vue';

const activeTab = ref('探索');

const navItems = [
  { label: '探索', icon: Sparkles, to: '/agents.view'},
  { label: '工作室', icon: LayoutGrid, to: '/workflow.view'},
  { label: '知识库', icon: BookOpen, to: '/workflow.view'},
  { label: '工具', icon: Hammer, to: '/workflow.view'},
];
</script>
