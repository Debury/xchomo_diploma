<template>
  <div class="max-w-2xl mx-auto">
    <div class="mb-6">
      <router-link to="/sources" class="text-gray-400 hover:text-white transition-colors">
        ← Back to Sources
      </router-link>
    </div>

    <div class="card">
      <h1 class="text-2xl font-bold text-white mb-2">Add New Source</h1>
      <p class="text-gray-400 mb-6">Configure a new climate data source</p>

      <form @submit.prevent="createSource" class="space-y-6">
        <!-- Source Name -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Source Name</label>
          <input
            v-model="form.name"
            type="text"
            class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            placeholder="e.g., ISIMIP, ERA5, CMIP6"
            required
          />
        </div>

        <!-- Source Type -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Data Type</label>
          <select
            v-model="form.type"
            class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
          >
            <option value="netcdf">NetCDF Files</option>
            <option value="csv">CSV Files</option>
            <option value="api">REST API</option>
            <option value="geotiff">GeoTIFF</option>
          </select>
        </div>

        <!-- Base URL -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Data URL / Path</label>
          <input
            v-model="form.url"
            type="text"
            class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            placeholder="https://data.isimip.org or /data/climate/"
          />
        </div>

        <!-- Variables -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Variables (comma separated)</label>
          <input
            v-model="form.variables"
            type="text"
            class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            placeholder="tas, pr, hurs, sfcWind"
          />
          <p class="mt-1 text-sm text-gray-500">Climate variables to extract from the dataset</p>
        </div>

        <!-- Temporal Range -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">Start Year</label>
            <input
              v-model.number="form.startYear"
              type="number"
              min="1900"
              max="2100"
              class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
              placeholder="2020"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-300 mb-2">End Year</label>
            <input
              v-model.number="form.endYear"
              type="number"
              min="1900"
              max="2100"
              class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
              placeholder="2100"
            />
          </div>
        </div>

        <!-- Description -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Description</label>
          <textarea
            v-model="form.description"
            rows="3"
            class="w-full bg-dark-hover border border-dark-border rounded-lg px-4 py-3 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500"
            placeholder="Brief description of this data source..."
          ></textarea>
        </div>

        <!-- Processing Options -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-3">Processing Options</label>
          <div class="space-y-3">
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.autoEmbed" type="checkbox" class="form-checkbox">
              <span class="text-gray-300">Auto-generate embeddings after import</span>
            </label>
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.enableSchedule" type="checkbox" class="form-checkbox">
              <span class="text-gray-300">Enable scheduled updates</span>
            </label>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex space-x-4 pt-4">
          <button
            type="submit"
            :disabled="submitting"
            class="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {{ submitting ? 'Creating...' : 'Create Source' }}
          </button>
          <router-link
            to="/sources"
            class="px-6 py-3 bg-dark-hover text-gray-300 rounded-lg hover:bg-gray-600 transition-colors text-center"
          >
            Cancel
          </router-link>
        </div>
      </form>
    </div>

    <!-- Help Section -->
    <div class="card mt-6">
      <h3 class="text-lg font-semibold text-white mb-4">Supported Data Sources</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div class="p-3 bg-dark-hover rounded">
          <h4 class="font-medium text-blue-400 mb-1">ISIMIP</h4>
          <p class="text-gray-400">Inter-Sectoral Impact Model Intercomparison Project data</p>
        </div>
        <div class="p-3 bg-dark-hover rounded">
          <h4 class="font-medium text-blue-400 mb-1">ERA5</h4>
          <p class="text-gray-400">ECMWF Reanalysis v5 climate data</p>
        </div>
        <div class="p-3 bg-dark-hover rounded">
          <h4 class="font-medium text-blue-400 mb-1">CMIP6</h4>
          <p class="text-gray-400">Coupled Model Intercomparison Project Phase 6</p>
        </div>
        <div class="p-3 bg-dark-hover rounded">
          <h4 class="font-medium text-blue-400 mb-1">Custom</h4>
          <p class="text-gray-400">Any NetCDF, CSV, or GeoTIFF files</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const form = ref({
  name: '',
  type: 'netcdf',
  url: '',
  variables: '',
  startYear: null,
  endYear: null,
  description: '',
  autoEmbed: true,
  enableSchedule: false
})

const submitting = ref(false)

async function createSource() {
  submitting.value = true
  
  try {
    // Validate required fields
    if (!form.value.name || !form.value.url) {
      alert('Please fill in Source Name and Data URL')
      return
    }

    // Map form fields to API fields
    const sourceConfig = {
      source_id: form.value.name.trim().toLowerCase().replace(/\s+/g, '_'), // Convert name to source_id
      url: form.value.url.trim(),
      format: form.value.type || null, // Map type to format
      variables: form.value.variables 
        ? form.value.variables.split(',').map(v => v.trim()).filter(Boolean)
        : null,
      description: form.value.description || null,
      tags: form.value.type ? [form.value.type] : null,
      time_range: (form.value.startYear || form.value.endYear) 
        ? {
            ...(form.value.startYear && { start: `${form.value.startYear}-01-01` }),
            ...(form.value.endYear && { end: `${form.value.endYear}-12-31` })
          }
        : null,
      is_active: true,
      embedding_model: "all-MiniLM-L6-v2"
    }

    // Remove null/empty fields
    Object.keys(sourceConfig).forEach(key => {
      if (sourceConfig[key] === null || sourceConfig[key] === undefined || 
          (Array.isArray(sourceConfig[key]) && sourceConfig[key].length === 0)) {
        delete sourceConfig[key]
      }
    })

    // Clean up time_range if empty
    if (sourceConfig.time_range && Object.keys(sourceConfig.time_range).length === 0) {
      delete sourceConfig.time_range
    }
    
    console.log('Creating source:', sourceConfig)
    
    // Call the actual API
    const resp = await fetch('/sources', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(sourceConfig)
    })
    
    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(errorData.detail || `HTTP ${resp.status}: ${resp.statusText}`)
    }
    
    const result = await resp.json()
    console.log('Source created:', result)
    
    alert(`Source "${form.value.name}" created successfully!`)
    router.push('/sources')
  } catch (e) {
    console.error('Error creating source:', e)
    alert(`Error creating source: ${e.message}`)
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.form-checkbox {
  @apply w-5 h-5 rounded border-gray-600 bg-dark-hover text-blue-600 focus:ring-blue-500 focus:ring-offset-0;
}
</style>
