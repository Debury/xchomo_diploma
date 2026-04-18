<template>
  <router-view />
  <ToastContainer />
  <ConfirmDialog />
</template>

<script setup lang="ts">
import ToastContainer from './components/ToastContainer.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
</script>
