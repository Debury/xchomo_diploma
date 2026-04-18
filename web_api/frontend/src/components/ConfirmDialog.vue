<template>
  <Modal
    :open="state.open"
    :title="state.title || 'Please confirm'"
    max-width="md"
    :scrollable="false"
    @close="cancel"
  >
    <p class="text-sm text-mendelu-black whitespace-pre-line">{{ state.message }}</p>
    <template #footer>
      <div class="flex justify-end gap-2 pt-4">
        <button type="button" class="btn-ghost" @click="cancel">{{ state.cancelText }}</button>
        <button
          type="button"
          :class="state.danger ? 'btn-danger' : 'btn-primary'"
          @click="accept"
        >
          {{ state.confirmText }}
        </button>
      </div>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import Modal from './Modal.vue'
import { useConfirm } from '../composables/useConfirm'

const { state, accept, cancel } = useConfirm()
</script>
