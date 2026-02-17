<template>
  <div class="max-w-2xl mx-auto">
    <div class="mb-6">
      <router-link to="/sources" class="text-mendelu-gray-dark hover:text-mendelu-green transition-colors text-sm">
        &larr; Back to Sources
      </router-link>
    </div>

    <div class="card">
      <h1 class="text-xl font-bold text-mendelu-black mb-2">Add New Source</h1>
      <p class="text-mendelu-gray-dark text-sm mb-6">Configure a new climate data source</p>

      <form @submit.prevent="createSource" class="space-y-5">
        <div>
          <label class="block text-sm font-medium text-mendelu-black mb-1">Source Name</label>
          <input v-model="form.name" type="text" class="input-field" placeholder="e.g., ISIMIP, ERA5, CMIP6" required />
        </div>

        <div>
          <label class="block text-sm font-medium text-mendelu-black mb-1">Data Type</label>
          <select v-model="form.type" class="input-field">
            <option value="netcdf">NetCDF Files</option>
            <option value="csv">CSV Files</option>
            <option value="api">REST API</option>
            <option value="geotiff">GeoTIFF</option>
          </select>
        </div>

        <div>
          <label class="block text-sm font-medium text-mendelu-black mb-1">Data URL / Path</label>
          <input v-model="form.url" type="text" class="input-field" placeholder="https://data.isimip.org or /data/climate/" />
        </div>

        <!-- Authentication -->
        <div class="p-4 bg-mendelu-gray-light rounded-lg space-y-4">
          <h3 class="text-sm font-semibold text-mendelu-black">Authentication</h3>

          <div>
            <label class="block text-sm font-medium text-mendelu-black mb-1">Portal Preset</label>
            <select v-model="form.portal" @change="onPortalChange" class="input-field">
              <option value="">Custom / None</option>
              <option value="CDS">Copernicus CDS</option>
              <option value="NASA">NASA Earthdata</option>
              <option value="MARINE">Marine Copernicus</option>
              <option value="ESGF">ESGF (CMIP6/CORDEX)</option>
              <option value="NOAA">NOAA PSL</option>
            </select>
            <p v-if="form.portal" class="mt-1 text-xs text-mendelu-green">Uses global credentials from Settings page</p>
          </div>

          <div>
            <label class="block text-sm font-medium text-mendelu-black mb-1">Authentication Method</label>
            <select v-model="form.auth_method" class="input-field">
              <option value="none">None (Open Access)</option>
              <option value="api_key">API Key</option>
              <option value="bearer_token">Bearer Token</option>
              <option value="basic">Username &amp; Password</option>
            </select>
          </div>

          <div v-if="form.auth_method === 'api_key'">
            <label class="block text-sm font-medium text-mendelu-black mb-1">API Key</label>
            <input type="password" v-model="form.auth_api_key" class="input-field" placeholder="Enter API key" />
          </div>
          <div v-if="form.auth_method === 'bearer_token'">
            <label class="block text-sm font-medium text-mendelu-black mb-1">Bearer Token</label>
            <input type="password" v-model="form.auth_token" class="input-field" placeholder="Enter bearer token" />
          </div>
          <div v-if="form.auth_method === 'basic'" class="space-y-3">
            <div>
              <label class="block text-sm font-medium text-mendelu-black mb-1">Username</label>
              <input type="text" v-model="form.auth_username" class="input-field" placeholder="Username" />
            </div>
            <div>
              <label class="block text-sm font-medium text-mendelu-black mb-1">Password</label>
              <input type="password" v-model="form.auth_password" class="input-field" placeholder="Password" />
            </div>
          </div>
        </div>

        <div>
          <label class="block text-sm font-medium text-mendelu-black mb-1">Variables (comma separated)</label>
          <input v-model="form.variables" type="text" class="input-field" placeholder="tas, pr, hurs, sfcWind" />
          <p class="mt-1 text-xs text-mendelu-gray-dark">Climate variables to extract from the dataset</p>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-sm font-medium text-mendelu-black mb-1">Start Year</label>
            <input v-model.number="form.startYear" type="number" min="1900" max="2100" class="input-field" placeholder="2020" />
          </div>
          <div>
            <label class="block text-sm font-medium text-mendelu-black mb-1">End Year</label>
            <input v-model.number="form.endYear" type="number" min="1900" max="2100" class="input-field" placeholder="2100" />
          </div>
        </div>

        <div>
          <label class="block text-sm font-medium text-mendelu-black mb-1">Description</label>
          <textarea v-model="form.description" rows="3" class="input-field resize-none" placeholder="Brief description of this data source..."></textarea>
        </div>

        <div>
          <label class="block text-sm font-medium text-mendelu-black mb-2">Processing Options</label>
          <div class="space-y-3">
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.autoEmbed" type="checkbox" class="form-checkbox">
              <span class="text-sm text-mendelu-black">Auto-generate embeddings after import</span>
            </label>
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.enableSchedule" type="checkbox" class="form-checkbox">
              <span class="text-sm text-mendelu-black">Enable scheduled updates</span>
            </label>
          </div>
        </div>

        <div class="flex space-x-4 pt-4">
          <button type="submit" :disabled="submitting" class="flex-1 btn-primary py-3 disabled:opacity-50">
            {{ submitting ? 'Creating...' : 'Create Source' }}
          </button>
          <router-link to="/sources" class="btn-secondary py-3 text-center flex-shrink-0 px-8">
            Cancel
          </router-link>
        </div>
      </form>
    </div>

    <div class="card mt-6">
      <h3 class="text-base font-semibold text-mendelu-black mb-4">Supported Data Sources</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div class="p-3 bg-mendelu-gray-light rounded-lg">
          <h4 class="font-medium text-mendelu-green mb-1">ISIMIP</h4>
          <p class="text-mendelu-gray-dark">Inter-Sectoral Impact Model Intercomparison Project data</p>
        </div>
        <div class="p-3 bg-mendelu-gray-light rounded-lg">
          <h4 class="font-medium text-mendelu-green mb-1">ERA5</h4>
          <p class="text-mendelu-gray-dark">ECMWF Reanalysis v5 climate data</p>
        </div>
        <div class="p-3 bg-mendelu-gray-light rounded-lg">
          <h4 class="font-medium text-mendelu-green mb-1">CMIP6</h4>
          <p class="text-mendelu-gray-dark">Coupled Model Intercomparison Project Phase 6</p>
        </div>
        <div class="p-3 bg-mendelu-gray-light rounded-lg">
          <h4 class="font-medium text-mendelu-green mb-1">Custom</h4>
          <p class="text-mendelu-gray-dark">Any NetCDF, CSV, or GeoTIFF files</p>
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
  name: '', type: 'netcdf', url: '', variables: '',
  startYear: null, endYear: null, description: '',
  autoEmbed: true, enableSchedule: false,
  auth_method: 'none', auth_api_key: '', auth_token: '',
  auth_username: '', auth_password: '', portal: ''
})

const PORTAL_AUTH_MAP = {
  CDS: 'api_key',
  NASA: 'bearer_token',
  MARINE: 'basic',
  ESGF: 'api_key',
  NOAA: 'none'
}

function onPortalChange() {
  if (form.value.portal && PORTAL_AUTH_MAP[form.value.portal]) {
    form.value.auth_method = PORTAL_AUTH_MAP[form.value.portal]
  }
}

const submitting = ref(false)

async function createSource() {
  submitting.value = true
  try {
    if (!form.value.name || !form.value.url) {
      alert('Please fill in Source Name and Data URL')
      return
    }
    // Build auth credentials object
    let authCredentials = null
    if (form.value.auth_method === 'api_key' && form.value.auth_api_key) {
      authCredentials = { api_key: form.value.auth_api_key }
    } else if (form.value.auth_method === 'bearer_token' && form.value.auth_token) {
      authCredentials = { token: form.value.auth_token }
    } else if (form.value.auth_method === 'basic' && form.value.auth_username) {
      authCredentials = { username: form.value.auth_username, password: form.value.auth_password }
    }

    const sourceConfig = {
      source_id: form.value.name.trim().toLowerCase().replace(/\s+/g, '_'),
      url: form.value.url.trim(),
      format: form.value.type || null,
      variables: form.value.variables
        ? form.value.variables.split(',').map(v => v.trim()).filter(Boolean) : null,
      description: form.value.description || null,
      tags: form.value.type ? [form.value.type] : null,
      time_range: (form.value.startYear || form.value.endYear)
        ? {
            ...(form.value.startYear && { start: `${form.value.startYear}-01-01` }),
            ...(form.value.endYear && { end: `${form.value.endYear}-12-31` })
          } : null,
      is_active: true,
      embedding_model: "all-MiniLM-L6-v2",
      auth_method: form.value.auth_method !== 'none' ? form.value.auth_method : null,
      auth_credentials: authCredentials,
      portal: form.value.portal || null
    }
    Object.keys(sourceConfig).forEach(key => {
      if (sourceConfig[key] === null || sourceConfig[key] === undefined ||
          (Array.isArray(sourceConfig[key]) && sourceConfig[key].length === 0)) {
        delete sourceConfig[key]
      }
    })
    if (sourceConfig.time_range && Object.keys(sourceConfig.time_range).length === 0) delete sourceConfig.time_range

    const resp = await fetch('/sources', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sourceConfig)
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
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
  @apply w-5 h-5 rounded border-mendelu-gray-semi bg-white text-mendelu-green focus:ring-mendelu-green focus:ring-offset-0;
}
</style>
