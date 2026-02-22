<template>
  <div class="max-w-2xl mx-auto">
    <div class="mb-6">
      <router-link to="/sources" class="btn-ghost !px-0 text-mendelu-gray-dark hover:text-mendelu-green">
        &larr; Back to Sources
      </router-link>
    </div>

    <div class="card">
      <h1 class="page-title mb-2">Add New Source</h1>

      <!-- Step indicator -->
      <div class="flex items-center gap-1 mb-6">
        <template v-for="s in totalSteps" :key="s">
          <div class="flex flex-col items-center gap-1" :class="s < totalSteps ? 'flex-1' : ''">
            <div class="flex items-center w-full">
              <div
                class="w-8 h-1 rounded-full transition-all duration-150 flex-1"
                :class="s <= step ? 'bg-mendelu-green' : 'bg-mendelu-gray-semi'"
              ></div>
            </div>
            <span class="text-[10px] text-mendelu-gray-dark">{{ stepLabels[s - 1] }}</span>
          </div>
        </template>
        <span class="ml-2 text-xs text-mendelu-gray-dark tabular-nums">{{ step }}/{{ totalSteps }}</span>
      </div>

      <form @submit.prevent="handleSubmit">
        <!-- Step 1: URL & Format -->
        <div v-show="step === 1" class="space-y-5">
          <p class="page-subtitle">Enter the data source URL</p>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Source Name</label>
            <input v-model="form.name" type="text" class="input-field" placeholder="e.g., ERA5, CMIP6" required />
          </div>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Data URL / Path</label>
            <div class="flex gap-2">
              <input v-model="form.url" type="text" class="input-field flex-1" placeholder="https://..." />
              <button
                type="button"
                @click="analyzeUrl"
                :disabled="!form.url || analyzing"
                class="btn-secondary !py-2 !px-4 !text-xs disabled:opacity-50 whitespace-nowrap"
              >
                {{ analyzing ? 'Analyzing...' : 'Auto-detect' }}
              </button>
            </div>
          </div>

          <div v-if="urlAnalysis" class="bg-mendelu-gray-light p-3 rounded-lg text-xs space-y-1">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-2 h-2 rounded-full" :class="urlAnalysis.reachable ? 'bg-mendelu-success' : 'bg-mendelu-alert'"></span>
              <span class="font-medium">{{ urlAnalysis.reachable ? 'URL reachable' : 'URL unreachable' }}</span>
              <span v-if="urlAnalysis.latency_ms" class="text-mendelu-gray-dark">({{ urlAnalysis.latency_ms }}ms)</span>
            </div>
            <p v-if="urlAnalysis.format"><span class="text-mendelu-gray-dark">Format:</span> {{ urlAnalysis.format }}</p>
            <p v-if="urlAnalysis.portal"><span class="text-mendelu-gray-dark">Portal:</span> {{ urlAnalysis.portal }}</p>
            <p v-if="urlAnalysis.suggested_auth"><span class="text-mendelu-gray-dark">Auth:</span> {{ urlAnalysis.suggested_auth }}</p>
            <p v-if="urlAnalysis.error" class="text-mendelu-alert">{{ urlAnalysis.error }}</p>
          </div>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Data Type</label>
            <select v-model="form.type" class="input-field">
              <option value="netcdf">NetCDF Files</option>
              <option value="csv">CSV Files</option>
              <option value="api">REST API</option>
              <option value="geotiff">GeoTIFF</option>
            </select>
          </div>
        </div>

        <!-- Step 2: Auth -->
        <div v-show="step === 2" class="space-y-5">
          <p class="page-subtitle">Configure authentication (optional)</p>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Portal Preset</label>
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
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Authentication Method</label>
            <select v-model="form.auth_method" class="input-field">
              <option value="none">None (Open Access)</option>
              <option value="api_key">API Key</option>
              <option value="bearer_token">Bearer Token</option>
              <option value="basic">Username &amp; Password</option>
            </select>
          </div>

          <div v-if="form.auth_method === 'api_key'">
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">API Key</label>
            <input type="password" v-model="form.auth_api_key" class="input-field" placeholder="Enter API key" />
          </div>
          <div v-if="form.auth_method === 'bearer_token'">
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Bearer Token</label>
            <input type="password" v-model="form.auth_token" class="input-field" placeholder="Enter bearer token" />
          </div>
          <div v-if="form.auth_method === 'basic'" class="space-y-3">
            <div>
              <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Username</label>
              <input type="text" v-model="form.auth_username" class="input-field" />
            </div>
            <div>
              <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Password</label>
              <input type="password" v-model="form.auth_password" class="input-field" />
            </div>
          </div>
        </div>

        <!-- Step 3: Variables & Config -->
        <div v-show="step === 3" class="space-y-5">
          <p class="page-subtitle">Configure data variables and time range</p>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Variables (comma separated)</label>
            <input v-model="form.variables" type="text" class="input-field" placeholder="tas, pr, hurs, sfcWind" />
            <p class="mt-1 text-xs text-mendelu-gray-dark">Climate variables to extract from the dataset</p>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Start Year</label>
              <input v-model.number="form.startYear" type="number" min="1900" max="2100" class="input-field" placeholder="2020" />
            </div>
            <div>
              <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">End Year</label>
              <input v-model.number="form.endYear" type="number" min="1900" max="2100" class="input-field" placeholder="2100" />
            </div>
          </div>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Description</label>
            <textarea v-model="form.description" rows="3" class="input-field resize-none" placeholder="Brief description..."></textarea>
          </div>
        </div>

        <!-- Step 4: Schedule -->
        <div v-show="step === 4" class="space-y-5">
          <p class="page-subtitle">Set up automatic re-processing schedule (optional)</p>

          <div class="space-y-3">
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.enableSchedule" type="checkbox" class="form-checkbox">
              <span class="text-sm text-mendelu-black">Enable scheduled updates</span>
            </label>
          </div>

          <div v-if="form.enableSchedule">
            <CronPicker v-model="form.schedule_cron" label="Update Schedule" />
          </div>

          <div class="space-y-3">
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.autoEmbed" type="checkbox" class="form-checkbox">
              <span class="text-sm text-mendelu-black">Auto-generate embeddings after import</span>
            </label>
          </div>
        </div>

        <!-- Step 5: Review -->
        <div v-show="step === 5" class="space-y-5">
          <p class="page-subtitle">Review your source configuration</p>

          <div class="bg-mendelu-gray-light p-4 rounded-lg text-sm space-y-2">
            <div class="flex justify-between"><span class="text-mendelu-gray-dark">Name</span><span class="text-mendelu-black font-medium">{{ form.name }}</span></div>
            <div class="flex justify-between"><span class="text-mendelu-gray-dark">URL</span><span class="text-mendelu-black text-xs break-all">{{ form.url }}</span></div>
            <div class="flex justify-between"><span class="text-mendelu-gray-dark">Format</span><span class="text-mendelu-black">{{ form.type }}</span></div>
            <div v-if="form.portal" class="flex justify-between"><span class="text-mendelu-gray-dark">Portal</span><span class="text-mendelu-black">{{ form.portal }}</span></div>
            <div v-if="form.auth_method !== 'none'" class="flex justify-between"><span class="text-mendelu-gray-dark">Auth</span><span class="text-mendelu-black">{{ form.auth_method }}</span></div>
            <div v-if="form.variables" class="flex justify-between"><span class="text-mendelu-gray-dark">Variables</span><span class="text-mendelu-black">{{ form.variables }}</span></div>
            <div v-if="form.enableSchedule" class="flex justify-between"><span class="text-mendelu-gray-dark">Schedule</span><code class="text-mendelu-black font-mono text-xs">{{ form.schedule_cron }}</code></div>
          </div>

          <button
            type="button"
            @click="testConnectionReview"
            :disabled="testingConnection"
            class="btn-secondary w-full disabled:opacity-50"
          >
            {{ testingConnection ? 'Testing...' : 'Test Connection' }}
          </button>
          <div v-if="reviewConnectionResult" class="p-3 rounded-lg text-xs" :class="reviewConnectionResult.reachable ? 'bg-mendelu-success/10 text-mendelu-success' : 'bg-mendelu-alert/10 text-mendelu-alert'">
            <strong>{{ reviewConnectionResult.reachable ? 'Connection OK' : 'Connection Failed' }}</strong>
            <span v-if="reviewConnectionResult.latency_ms"> ({{ reviewConnectionResult.latency_ms }}ms)</span>
            <p v-if="reviewConnectionResult.error">{{ reviewConnectionResult.error }}</p>
          </div>
        </div>

        <!-- Navigation buttons -->
        <div class="flex gap-3 pt-6">
          <button v-if="step > 1" type="button" @click="step--" class="btn-secondary flex-1 py-3">Back</button>
          <button v-if="step < totalSteps" type="button" @click="nextStep" class="btn-primary flex-1 py-3">Next</button>
          <button v-if="step === totalSteps" type="submit" :disabled="submitting" class="btn-primary flex-1 py-3 disabled:opacity-50">
            {{ submitting ? 'Creating...' : 'Create Source' }}
          </button>
          <router-link v-if="step === 1" to="/sources" class="btn-secondary py-3 text-center flex-shrink-0 px-8">Cancel</router-link>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import CronPicker from '../components/CronPicker.vue'

const router = useRouter()

const step = ref(1)
const totalSteps = 5
const stepLabels = ['URL', 'Auth', 'Config', 'Schedule', 'Review']

const form = ref({
  name: '', type: 'netcdf', url: '', variables: '',
  startYear: null, endYear: null, description: '',
  autoEmbed: true, enableSchedule: false, schedule_cron: '0 2 * * 0',
  auth_method: 'none', auth_api_key: '', auth_token: '',
  auth_username: '', auth_password: '', portal: ''
})

const submitting = ref(false)
const analyzing = ref(false)
const urlAnalysis = ref(null)
const testingConnection = ref(false)
const reviewConnectionResult = ref(null)

const PORTAL_AUTH_MAP = {
  CDS: 'api_key', NASA: 'bearer_token', MARINE: 'basic', ESGF: 'api_key', NOAA: 'none'
}

function onPortalChange() {
  if (form.value.portal && PORTAL_AUTH_MAP[form.value.portal]) {
    form.value.auth_method = PORTAL_AUTH_MAP[form.value.portal]
  }
}

function nextStep() {
  if (step.value === 1 && (!form.value.name || !form.value.url)) {
    alert('Please fill in Source Name and Data URL')
    return
  }
  step.value++
}

async function analyzeUrl() {
  if (!form.value.url) return
  analyzing.value = true
  urlAnalysis.value = null
  try {
    const resp = await fetch('/sources/analyze-url', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: form.value.url }),
    })
    if (resp.ok) {
      const data = await resp.json()
      urlAnalysis.value = data
      if (data.format) form.value.type = data.format
      if (data.portal) { form.value.portal = data.portal; onPortalChange() }
    }
  } catch (e) {
    urlAnalysis.value = { reachable: false, error: e.message }
  } finally {
    analyzing.value = false
  }
}

async function testConnectionReview() {
  testingConnection.value = true
  reviewConnectionResult.value = null
  try {
    const resp = await fetch('/sources/analyze-url', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: form.value.url }),
    })
    if (resp.ok) reviewConnectionResult.value = await resp.json()
  } catch (e) {
    reviewConnectionResult.value = { reachable: false, error: e.message }
  } finally {
    testingConnection.value = false
  }
}

async function handleSubmit() {
  submitting.value = true
  try {
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
      variables: form.value.variables ? form.value.variables.split(',').map(v => v.trim()).filter(Boolean) : null,
      description: form.value.description || null,
      tags: form.value.type ? [form.value.type] : null,
      time_range: (form.value.startYear || form.value.endYear)
        ? { ...(form.value.startYear && { start: `${form.value.startYear}-01-01` }), ...(form.value.endYear && { end: `${form.value.endYear}-12-31` }) } : null,
      is_active: true,
      embedding_model: "all-MiniLM-L6-v2",
      auth_method: form.value.auth_method !== 'none' ? form.value.auth_method : null,
      auth_credentials: authCredentials,
      portal: form.value.portal || null,
      schedule_cron: form.value.enableSchedule ? form.value.schedule_cron : null,
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
