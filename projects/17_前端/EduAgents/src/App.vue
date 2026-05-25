<template>
  <div class="min-h-screen bg-gray-50 font-sans text-gray-900 antialiased">
    <AppHeader />
    <router-view/>   <!-- 该区域使用配置的路由来替换 -->
  </div>
</template>

<script setup>
import AppHeader from './components/Header.vue';   // 组件
import { useRouter } from 'vue-router'; // 引入路由
// 创建 路由器 对象
const router = useRouter();
// 强制显示指定的路由
router.push('/agents.view');
</script>

