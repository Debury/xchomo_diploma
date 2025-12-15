import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from './stores/auth'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('./views/Login.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/',
    component: () => import('./layouts/MainLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        name: 'Dashboard',
        component: () => import('./views/Dashboard.vue')
      },
      {
        path: 'chat',
        name: 'Chat',
        component: () => import('./views/Chat.vue')
      },
      {
        path: 'sources',
        name: 'Sources',
        component: () => import('./views/Sources.vue')
      },
      {
        path: 'sources/create',
        name: 'CreateSource',
        component: () => import('./views/CreateSource.vue')
      },
      {
        path: 'embeddings',
        name: 'Embeddings',
        component: () => import('./views/Embeddings.vue')
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory('/app/'),
  routes
})

// Navigation guard
router.beforeEach((to, from, next) => {
  const authStore = useAuthStore()
  
  if (to.meta.requiresAuth !== false && !authStore.isAuthenticated) {
    next({ name: 'Login' })
  } else if (to.name === 'Login' && authStore.isAuthenticated) {
    next({ name: 'Dashboard' })
  } else {
    next()
  }
})

export default router
