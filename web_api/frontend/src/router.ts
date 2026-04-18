import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useAuthStore } from './stores/auth'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('./views/Login.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/',
    component: () => import('./layouts/MainLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      { path: '', name: 'Dashboard', component: () => import('./views/Dashboard.vue') },
      { path: 'chat', name: 'Chat', component: () => import('./views/Chat.vue') },
      { path: 'sources', name: 'Sources', component: () => import('./views/Sources.vue') },
      { path: 'sources/create', name: 'CreateSource', component: () => import('./views/CreateSource.vue') },
      { path: 'catalog', name: 'Catalog', component: () => import('./views/Catalog.vue') },
      { path: 'etl', name: 'ETL Monitor', component: () => import('./views/ETLMonitor.vue') },
      { path: 'schedules', name: 'Schedules', component: () => import('./views/Schedules.vue') },
      { path: 'settings', name: 'Settings', component: () => import('./views/Settings.vue') },
    ],
  },
]

const router = createRouter({
  history: createWebHistory('/app/'),
  routes,
})

router.beforeEach((to, _from, next) => {
  const authStore = useAuthStore()
  const requiresAuth = to.matched.some(r => r.meta?.requiresAuth === true)

  if (requiresAuth && !authStore.isAuthenticated) {
    next({
      name: 'Login',
      query: to.fullPath !== '/' ? { redirect: to.fullPath } : {},
    })
  } else if (to.name === 'Login' && authStore.isAuthenticated) {
    next({ name: 'Dashboard' })
  } else {
    next()
  }
})

export default router
