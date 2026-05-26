import { createRouter, createWebHistory } from 'vue-router'
import AgentsView from '../views/AgentsView.vue'
import WorkflowView from '../views/WorkflowView.vue'
import OutlineAgentView from '../views/OutlineAgentView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/agents.view',
    },
    {
      path: '/agents.view',
      name: 'agents',
      component: AgentsView,
    },
    {
      path: '/workflow.view',
      name: 'workflow',
      component: WorkflowView
    },
    {
      path: '/outline.view',
      name: 'outline',
      component: OutlineAgentView
    }
  ],
})

export default router
