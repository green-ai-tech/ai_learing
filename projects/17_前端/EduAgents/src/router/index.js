import { createRouter, createWebHistory } from 'vue-router'
import AgentsView from '../views/AgentsView.vue'
import WorkflowView from '../views/WorkflowView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/agents.view',
      name: 'agents',
      component: AgentsView,
    },
    {
      path: '/workflow.view',
      name: 'workflow',
      component: WorkflowView
    }
  ],
})

export default router
