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
            <label for="cs-name" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Source Name</label>
            <input id="cs-name" v-model="form.name" type="text" class="input-field" placeholder="e.g., ERA5, CMIP6" required />
          </div>

          <div>
            <label class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Data Source</label>
            <div class="inline-flex rounded-lg border border-mendelu-gray-semi overflow-hidden mb-3 text-xs">
              <button
                type="button"
                @click="sourceMode = 'url'"
                :class="sourceMode === 'url' ? 'bg-mendelu-green text-white' : 'bg-white text-mendelu-gray-dark hover:bg-mendelu-gray-light'"
                class="px-4 py-1.5 font-medium transition-colors"
              >URL / link</button>
              <button
                type="button"
                @click="sourceMode = 'upload'"
                :class="sourceMode === 'upload' ? 'bg-mendelu-green text-white' : 'bg-white text-mendelu-gray-dark hover:bg-mendelu-gray-light'"
                class="px-4 py-1.5 font-medium transition-colors border-l border-mendelu-gray-semi"
              >Upload file</button>
            </div>

            <!-- URL mode -->
            <div v-if="sourceMode === 'url'">
              <div class="flex gap-2">
                <input id="cs-url" v-model="form.url" type="text" class="input-field flex-1" placeholder="https://..." />
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

            <!-- Upload mode -->
            <div v-else class="space-y-2">
              <div class="flex items-center gap-2">
                <label
                  for="cs-file"
                  class="btn-secondary !py-2 !px-4 !text-xs cursor-pointer whitespace-nowrap"
                  :class="{ 'opacity-50 cursor-wait': uploading }"
                >
                  {{ uploading ? 'Uploading…' : (uploadedFile ? 'Choose another file' : 'Choose file…') }}
                </label>
                <input
                  id="cs-file"
                  type="file"
                  class="hidden"
                  accept=".nc,.nc4,.cdf,.hdf,.hdf5,.h5,.he5,.tif,.tiff,.grib,.grib2,.grb,.grb2,.csv,.tsv,.txt,.zip,.gz,.tar,.zarr,.parquet"
                  @change="handleFileUpload"
                  :disabled="uploading"
                />
                <span v-if="uploadedFile" class="text-xs text-mendelu-gray-dark truncate">
                  <span class="text-mendelu-success font-medium">✓</span>
                  {{ uploadedFile.filename }}
                  <span class="text-mendelu-gray-dark">({{ formatBytes(uploadedFile.size_bytes) }})</span>
                </span>
              </div>
              <p class="text-xs text-mendelu-gray-dark">
                Max {{ uploadMaxMb }} MB per file
                (<span class="font-mono">UPLOAD_MAX_MB</span> env var).
                Accepted: NetCDF, GRIB, HDF5, GeoTIFF, CSV, Parquet, Zarr, zip/gz/tar.
                The file stays on the server and the pipeline reads it in place.
              </p>
              <p v-if="uploadError" class="text-xs text-mendelu-alert">{{ uploadError }}</p>
              <!-- Mirror of the path stored in form.url, shown so the user
                   knows what URL will be sent with the create request. -->
              <div v-if="uploadedFile" class="text-[11px] font-mono text-mendelu-gray-dark break-all">
                path: {{ form.url }}
              </div>
            </div>
          </div>

          <div v-if="urlAnalysis" class="bg-mendelu-gray-light p-3 rounded-lg text-xs space-y-2">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full" :class="urlAnalysis.reachable ? 'bg-mendelu-success' : 'bg-mendelu-alert'"></span>
              <span class="font-medium">{{ urlAnalysis.reachable ? 'URL reachable' : 'URL unreachable' }}</span>
              <span v-if="urlAnalysis.latency_ms > 0" class="text-mendelu-gray-dark">({{ urlAnalysis.latency_ms }}ms)</span>
            </div>
            <div class="grid grid-cols-2 gap-x-4 gap-y-1">
              <p v-if="urlAnalysis.format"><span class="text-mendelu-gray-dark">Format:</span> <span class="font-medium">{{ urlAnalysis.format }}</span></p>
              <p v-if="urlAnalysis.portal"><span class="text-mendelu-gray-dark">Portal:</span> <span class="font-medium">{{ urlAnalysis.portal }}</span></p>
              <p v-if="urlAnalysis.suggested_auth"><span class="text-mendelu-gray-dark">Auth:</span> <span class="font-medium">{{ urlAnalysis.suggested_auth }}</span></p>
              <p v-if="urlAnalysis.suggested_name"><span class="text-mendelu-gray-dark">Name:</span> <span class="font-medium">{{ urlAnalysis.suggested_name }}</span></p>
            </div>
            <!-- Dataset grouping suggestion -->
            <div v-if="urlAnalysis.matched_dataset && urlAnalysis.reachable" class="mt-2 p-2 bg-mendelu-success/10 border border-mendelu-success/20 rounded-lg">
              <div class="flex items-center justify-between">
                <div>
                  <span class="font-medium text-mendelu-success">Existing dataset found:</span>
                  <span class="text-mendelu-black ml-1">{{ urlAnalysis.matched_dataset.dataset_name }}</span>
                  <span class="text-mendelu-gray-dark ml-1">({{ urlAnalysis.matched_dataset.chunk_count?.toLocaleString() }} chunks)</span>
                </div>
                <button type="button" @click="useMatchedDataset" class="text-mendelu-green font-medium hover:underline">
                  Add to this dataset
                </button>
              </div>
            </div>
            <div v-else-if="urlAnalysis.suggested_name && urlAnalysis.reachable && !urlAnalysis.matched_dataset" class="mt-2 p-2 bg-mendelu-gray-light border border-mendelu-gray-semi rounded-lg">
              <span class="text-mendelu-gray-dark">New dataset:</span>
              <span class="font-medium text-mendelu-black ml-1">{{ urlAnalysis.suggested_name }}</span>
              <button type="button" @click="useSuggestedName" class="ml-2 text-mendelu-green font-medium hover:underline">
                Use this name
              </button>
            </div>
            <p v-if="urlAnalysis.error" class="text-mendelu-alert">{{ urlAnalysis.error }}</p>
          </div>

          <div>
            <label for="cs-type" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Data Format</label>
            <select id="cs-type" v-model="form.type" class="input-field">
              <option value="netcdf">NetCDF (.nc, .nc4)</option>
              <option value="grib">GRIB (.grib, .grib2)</option>
              <option value="hdf5">HDF5 (.h5, .hdf, .he5)</option>
              <option value="geotiff">GeoTIFF (.tif, .tiff)</option>
              <option value="csv">CSV / TSV (.csv, .tsv, .txt)</option>
              <option value="zarr">Zarr (.zarr)</option>
              <option value="ascii">ASCII Grid (.asc)</option>
              <option value="zip">Archive (.zip, .gz, .tar)</option>
            </select>
          </div>
        </div>

        <!-- Step 2: Auth -->
        <div v-show="step === 2" class="space-y-5">
          <p class="page-subtitle">Configure authentication (optional)</p>

          <div>
            <label for="cs-portal" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Portal Preset</label>
            <select id="cs-portal" v-model="form.portal" @change="onPortalChange" class="input-field">
              <option value="">Custom / None</option>
              <option value="CDS">Copernicus CDS</option>
              <option value="NASA">NASA Earthdata</option>
              <option value="MARINE">Marine Copernicus</option>
              <option value="ESGF">ESGF (CMIP6/CORDEX)</option>
              <option value="NOAA">NOAA PSL</option>
            </select>
          </div>

          <!-- Global credentials detected -->
          <div v-if="form.portal && portalHasGlobalCreds" class="bg-mendelu-success/10 border border-mendelu-success/30 p-3 rounded-lg text-xs space-y-1">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-mendelu-success"></span>
              <span class="font-medium text-mendelu-success">Global {{ form.portal }} credentials configured</span>
            </div>
            <p class="text-mendelu-gray-dark">Will use credentials from Settings page. No extra setup needed.</p>
          </div>

          <!-- No global credentials for this portal -->
          <div v-if="form.portal && !portalHasGlobalCreds && PORTAL_CREDENTIAL_KEYS[form.portal]?.length" class="bg-mendelu-alert/10 border border-mendelu-alert/30 p-3 rounded-lg text-xs space-y-1">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-mendelu-alert"></span>
              <span class="font-medium text-mendelu-alert">{{ form.portal }} credentials not configured</span>
            </div>
            <p class="text-mendelu-gray-dark">Set up global credentials in <router-link to="/settings" class="underline text-mendelu-green">Settings</router-link>, or enter per-source credentials below.</p>
          </div>

          <!-- Manual auth (show when no portal or portal creds missing) -->
          <template v-if="!form.portal || !portalHasGlobalCreds">
            <div>
              <label for="cs-auth-method" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Authentication Method</label>
              <select id="cs-auth-method" v-model="form.auth_method" class="input-field">
                <option value="none">None (Open Access)</option>
                <option value="api_key">API Key</option>
                <option value="bearer_token">Bearer Token</option>
                <option value="basic">Username &amp; Password</option>
              </select>
            </div>

            <div v-if="form.auth_method === 'api_key'">
              <label for="cs-auth-api-key" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">API Key</label>
              <input id="cs-auth-api-key" type="password" v-model="form.auth_api_key" class="input-field" placeholder="Enter API key" />
            </div>
            <div v-if="form.auth_method === 'bearer_token'">
              <label for="cs-auth-token" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Bearer Token</label>
              <input id="cs-auth-token" type="password" v-model="form.auth_token" class="input-field" placeholder="Enter bearer token" />
            </div>
            <div v-if="form.auth_method === 'basic'" class="space-y-3">
              <div>
                <label for="cs-auth-username" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Username</label>
                <input id="cs-auth-username" type="text" v-model="form.auth_username" class="input-field" />
              </div>
              <div>
                <label for="cs-auth-password" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Password</label>
                <input id="cs-auth-password" type="password" v-model="form.auth_password" class="input-field" />
              </div>
            </div>
          </template>
        </div>

        <!-- Step 3: Variables & Metadata -->
        <div v-show="step === 3" class="space-y-5">
          <div class="flex items-center justify-between">
            <p class="page-subtitle">File metadata — auto-discovered from the source</p>
            <button
              type="button"
              @click="scanMetadata"
              :disabled="!form.url || scanning"
              class="btn-primary !py-2 !px-4 !text-xs disabled:opacity-50"
            >
              {{ scanning ? 'Scanning...' : 'Scan File' }}
            </button>
          </div>

          <!-- Scan results -->
          <div v-if="scanResult && !scanResult.error" class="bg-mendelu-success/5 border border-mendelu-success/20 p-4 rounded-lg space-y-3">
            <div class="flex items-center gap-2 text-xs font-medium text-mendelu-success">
              <span class="w-2 h-2 rounded-full bg-mendelu-success"></span>
              File scanned — metadata auto-filled
            </div>

            <div v-if="scanResult.variables?.length" class="text-xs">
              <span class="text-mendelu-gray-dark">Variables found:</span>
              <div class="flex flex-wrap gap-1 mt-1">
                <span v-for="v in scanResult.variables" :key="v.name" class="badge-info !text-[10px]">
                  {{ v.name }}<span v-if="v.long_name" class="text-mendelu-gray-dark ml-1">({{ v.long_name }})</span>
                </span>
              </div>
            </div>

            <div v-if="scanResult.time_range?.start" class="text-xs">
              <span class="text-mendelu-gray-dark">Time range:</span>
              <span class="ml-1">{{ scanResult.time_range.start }} — {{ scanResult.time_range.end }}</span>
            </div>

            <div v-if="scanResult.spatial_extent?.lat_min != null" class="text-xs">
              <span class="text-mendelu-gray-dark">Spatial:</span>
              <span class="ml-1">{{ scanResult.spatial_extent.lat_min.toFixed(1) }}° to {{ scanResult.spatial_extent.lat_max.toFixed(1) }}°N, {{ scanResult.spatial_extent.lon_min.toFixed(1) }}° to {{ scanResult.spatial_extent.lon_max.toFixed(1) }}°E</span>
            </div>

            <div v-if="Object.keys(scanResult.attributes || {}).length" class="text-xs">
              <span class="text-mendelu-gray-dark">Source:</span>
              <span v-if="scanResult.attributes.institution" class="ml-1">{{ scanResult.attributes.institution }}</span>
              <span v-if="scanResult.attributes.title" class="ml-1">— {{ scanResult.attributes.title }}</span>
            </div>
          </div>

          <div v-if="scanResult?.error" class="bg-mendelu-alert/5 border border-mendelu-alert/20 p-3 rounded-lg text-xs text-mendelu-alert">
            Could not scan file: {{ scanResult.error }}
          </div>

          <p v-if="!scanResult" class="text-xs text-mendelu-gray-dark">
            Click "Scan File" to auto-detect variables, time range, and spatial coverage from the file headers. You can also fill these in manually.
          </p>

          <!-- Manual overrides (collapsed by default if scan succeeded) -->
          <details :open="!scanResult || scanResult.error">
            <summary class="text-xs font-medium text-mendelu-gray-dark cursor-pointer hover:text-mendelu-black">
              {{ scanResult && !scanResult.error ? 'Edit metadata manually' : 'Manual metadata entry' }}
            </summary>
            <div class="mt-3 space-y-4">
              <div>
                <label for="cs-variables" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Variables (comma separated)</label>
                <input id="cs-variables" v-model="form.variables" type="text" class="input-field" placeholder="Auto-filled from scan" />
              </div>

              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label for="cs-start-year" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Start Year</label>
                  <input id="cs-start-year" v-model.number="form.startYear" type="number" min="1900" max="2100" class="input-field" />
                </div>
                <div>
                  <label for="cs-end-year" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">End Year</label>
                  <input id="cs-end-year" v-model.number="form.endYear" type="number" min="1900" max="2100" class="input-field" />
                </div>
              </div>

              <div>
                <label for="cs-hazard" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Hazard Type</label>
                <select id="cs-hazard" v-model="form.hazard_type" class="input-field">
                  <option value="">Not specified</option>
                  <option value="Drought">Drought</option>
                  <option value="Flood">Flood</option>
                  <option value="Heat">Heat</option>
                  <option value="Wildfire">Wildfire</option>
                  <option value="Storm">Storm</option>
                  <option value="Cold">Cold</option>
                  <option value="Multi-hazard">Multi-hazard</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <div>
                <label for="cs-region" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Region / Country</label>
                <input id="cs-region" v-model="form.region_country" type="text" class="input-field" placeholder="Auto-detected from coordinates" />
              </div>

              <div>
                <label for="cs-description" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Description</label>
                <textarea id="cs-description" v-model="form.description" rows="3" class="input-field resize-none" placeholder="Auto-generated from file metadata"></textarea>
              </div>

              <div>
                <label for="cs-keywords" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Keywords (comma separated)</label>
                <input id="cs-keywords" v-model="form.keywords" type="text" class="input-field" placeholder="Auto-generated from variables and attributes" />
              </div>

              <div>
                <label for="cs-custom-meta" class="block text-xs font-medium text-mendelu-gray-dark uppercase tracking-wider mb-1">Custom Metadata</label>
                <textarea id="cs-custom-meta" v-model="form.custom_metadata_raw" rows="2" class="input-field resize-none" placeholder='e.g., institution=ECMWF, project=Copernicus'></textarea>
              </div>
            </div>
          </details>
        </div>

        <!-- Step 4: Schedule -->
        <div v-show="step === 4" class="space-y-5">
          <p class="page-subtitle">Optionally schedule automatic re-processing for this source</p>

          <div class="space-y-3">
            <label class="flex items-center space-x-3 cursor-pointer">
              <input v-model="form.enableSchedule" type="checkbox" class="form-checkbox">
              <span class="text-sm text-mendelu-black">Enable per-source schedule</span>
            </label>
            <p class="text-xs text-mendelu-gray-dark">
              Each source has its own cron schedule. You can also manage schedules later on the Schedules page.
            </p>
          </div>

          <div v-if="form.enableSchedule">
            <CronPicker v-model="form.schedule_cron" label="Update Schedule" />
          </div>

          <div class="space-y-3 pt-3 border-t border-mendelu-gray-semi">
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
            <div v-if="form.hazard_type" class="flex justify-between"><span class="text-mendelu-gray-dark">Hazard</span><span class="text-mendelu-black">{{ form.hazard_type }}</span></div>
            <div v-if="form.region_country" class="flex justify-between"><span class="text-mendelu-gray-dark">Region</span><span class="text-mendelu-black">{{ form.region_country }}</span></div>
            <div v-if="form.spatial_coverage" class="flex justify-between"><span class="text-mendelu-gray-dark">Spatial</span><span class="text-mendelu-black">{{ form.spatial_coverage }}</span></div>
            <div v-if="form.impact_sector" class="flex justify-between"><span class="text-mendelu-gray-dark">Sector</span><span class="text-mendelu-black">{{ form.impact_sector }}</span></div>
            <div v-if="form.keywords" class="flex justify-between"><span class="text-mendelu-gray-dark">Keywords</span><span class="text-mendelu-black">{{ form.keywords }}</span></div>
            <div v-if="form.custom_metadata_raw" class="flex justify-between"><span class="text-mendelu-gray-dark">Custom Metadata</span><span class="text-mendelu-black text-xs">{{ form.custom_metadata_raw }}</span></div>
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

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import CronPicker from '../components/CronPicker.vue'
import { apiFetch } from '../api'
import { useToast } from '../composables/useToast'

const toast = useToast()

const router = useRouter()

const step = ref(1)
const totalSteps = 5
const stepLabels = ['URL', 'Auth', 'Config', 'Schedule', 'Review']

const form = ref<any>({
  name: '', type: 'netcdf', url: '', variables: '',
  startYear: null, endYear: null, description: '',
  autoEmbed: true, enableSchedule: false, schedule_cron: '0 2 * * 0',
  auth_method: 'none', auth_api_key: '', auth_token: '',
  auth_username: '', auth_password: '', portal: '',
  hazard_type: '', region_country: '', spatial_coverage: '', impact_sector: '',
  keywords: '', custom_metadata_raw: ''
})

const submitting = ref(false)
const analyzing = ref(false)
const urlAnalysis = ref(null)
const scanning = ref(false)
const scanResult = ref(null)
const testingConnection = ref(false)
const reviewConnectionResult = ref(null)

// URL vs local-file upload mode for the data source. When 'upload', the file
// is pushed to POST /sources/upload, saved under data/uploads/, and its
// server path is used as form.url — keeps the rest of the create flow
// untouched.
const sourceMode = ref<'url' | 'upload'>('url')
const uploading = ref(false)
const uploadedFile = ref<any>(null)
const uploadError = ref<string>('')
// Fetched from /settings/system so the hint below the file picker matches
// whatever the backend's UPLOAD_MAX_MB is actually set to. 5000 MB (5 GB) is
// the default, but operators can raise/lower it without a rebuild.
const uploadMaxMb = ref<number>(5000)

function formatBytes(n: number): string {
  if (!n) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let size = n
  while (size >= 1024 && i < units.length - 1) { size /= 1024; i++ }
  return `${size.toFixed(size >= 10 || i === 0 ? 0 : 1)} ${units[i]}`
}

async function handleFileUpload(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  uploadError.value = ''
  uploading.value = true
  try {
    const fd = new FormData()
    fd.append('file', file)
    const resp = await apiFetch('/sources/upload', { method: 'POST', body: fd })
    const body = await resp.json().catch(() => ({}))
    if (!resp.ok) {
      const msg = Array.isArray(body.detail)
        ? body.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
        : (body.detail || `HTTP ${resp.status}`)
      throw new Error(msg)
    }
    uploadedFile.value = body
    form.value.url = body.file_path
    // Format hint from the extension — makes the next step's defaults sensible.
    const ext = (body.filename || '').toLowerCase().split('.').pop()
    const fmtByExt: Record<string, string> = {
      nc: 'netcdf', nc4: 'netcdf', cdf: 'netcdf',
      h5: 'hdf5', hdf: 'hdf5', hdf5: 'hdf5', he5: 'hdf5',
      tif: 'geotiff', tiff: 'geotiff',
      grib: 'grib', grib2: 'grib', grb: 'grib', grb2: 'grib',
      csv: 'csv', tsv: 'csv', txt: 'csv',
      zip: 'zip', gz: 'zip', tar: 'zip',
      zarr: 'zarr',
    }
    if (ext && fmtByExt[ext] && !form.value.type) form.value.type = fmtByExt[ext]
    toast.success(`Uploaded ${body.filename} (${formatBytes(body.size_bytes)})`)
  } catch (e: any) {
    uploadError.value = e?.message || 'Upload failed'
    toast.error(`Upload failed: ${uploadError.value}`)
  } finally {
    uploading.value = false
    // Allow re-picking the same file after a failure.
    input.value = ''
  }
}

// Global credential status — fetched from Settings API
const globalCredentials = ref<any>({})

const PORTAL_AUTH_MAP = {
  CDS: 'api_key', NASA: 'basic', MARINE: 'basic', ESGF: 'none', NOAA: 'none'
}

// Which global credential keys each portal needs
const PORTAL_CREDENTIAL_KEYS = {
  CDS: ['cds_api_key'],
  NASA: ['nasa_earthdata_user', 'nasa_earthdata_password'],
  MARINE: ['cmems_username', 'cmems_password'],
  ESGF: [],
  NOAA: [],
}

// Check if global credentials are configured for the selected portal
const portalHasGlobalCreds = computed(() => {
  if (!form.value.portal) return false
  const keys = PORTAL_CREDENTIAL_KEYS[form.value.portal] || []
  if (keys.length === 0) return true  // No creds needed (ESGF, NOAA)
  return keys.every(k => globalCredentials.value[k]?.configured)
})

onMounted(async () => {
  try {
    const credResp = await apiFetch('/settings/credentials')
    if (credResp.ok) globalCredentials.value = await credResp.json()
  } catch (e) { /* ignore */ }
  try {
    const sysResp = await apiFetch('/settings/system')
    if (sysResp.ok) {
      const sys = await sysResp.json()
      if (sys?.uploads?.max_mb) uploadMaxMb.value = sys.uploads.max_mb
    }
  } catch (e) { /* ignore — keep default 5000 */ }
})

function useMatchedDataset() {
  if (urlAnalysis.value?.matched_dataset) {
    form.value.name = urlAnalysis.value.matched_dataset.dataset_name
  }
}

async function scanMetadata() {
  if (!form.value.url) return
  scanning.value = true
  scanResult.value = null
  try {
    const resp = await apiFetch('/sources/scan-metadata', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: form.value.url }),
    })
    if (resp.ok) {
      const data = await resp.json()
      scanResult.value = data
      if (!data.error) {
        // Auto-fill form fields
        if (data.variables?.length) {
          form.value.variables = data.variables.map(v => v.name).join(', ')
        }
        if (data.time_range?.start) {
          form.value.startYear = parseInt(data.time_range.start.substring(0, 4)) || null
        }
        if (data.time_range?.end) {
          form.value.endYear = parseInt(data.time_range.end.substring(0, 4)) || null
        }
        if (data.spatial_extent?.lat_min != null) {
          const ext = data.spatial_extent
          if (ext.lat_min < -60 && ext.lat_max > 60) form.value.region_country = 'Global'
          else if (ext.lat_min > 35 && ext.lat_max < 72 && ext.lon_min > -25 && ext.lon_max < 45) form.value.region_country = 'Europe'
          else form.value.region_country = `${ext.lat_min.toFixed(0)}N-${ext.lat_max.toFixed(0)}N, ${ext.lon_min.toFixed(0)}E-${ext.lon_max.toFixed(0)}E`
        }
        if (data.description) form.value.description = data.description
        // Generate keywords from variables and attributes
        const kw = []
        for (const v of data.variables || []) {
          if (v.long_name) kw.push(v.long_name)
        }
        if (data.attributes?.institution) kw.push(data.attributes.institution)
        if (kw.length) form.value.keywords = kw.join(', ')
        // Custom metadata from attributes
        const meta = []
        if (data.attributes?.institution) meta.push(`institution=${data.attributes.institution}`)
        if (data.attributes?.source) meta.push(`source=${data.attributes.source}`)
        if (meta.length) form.value.custom_metadata_raw = meta.join('\n')
      }
    }
  } catch (e) {
    scanResult.value = { error: e.message }
  } finally {
    scanning.value = false
  }
}

function useSuggestedName() {
  if (urlAnalysis.value?.suggested_name) {
    form.value.name = urlAnalysis.value.suggested_name
  }
}

function onPortalChange() {
  if (form.value.portal && PORTAL_AUTH_MAP[form.value.portal]) {
    form.value.auth_method = PORTAL_AUTH_MAP[form.value.portal]
  }
}

function nextStep() {
  if (step.value === 1 && (!form.value.name || !form.value.url)) {
    toast.error('Please fill in Source Name and Data URL')
    return
  }
  step.value++
}

async function analyzeUrl() {
  if (!form.value.url) return
  analyzing.value = true
  urlAnalysis.value = null
  try {
    const resp = await apiFetch('/sources/analyze-url', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: form.value.url }),
    })
    if (resp.ok) {
      const data = await resp.json()
      urlAnalysis.value = data
      if (data.format) form.value.type = data.format
      if (data.portal) { form.value.portal = data.portal; onPortalChange() }
      // Auto-fill name from matched dataset or suggestion
      if (!form.value.name) {
        if (data.matched_dataset) form.value.name = data.matched_dataset.dataset_name
        else if (data.suggested_name) form.value.name = data.suggested_name
      }
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
    const resp = await apiFetch('/sources/analyze-url', {
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
      embedding_model: "BAAI/bge-large-en-v1.5",
      auth_method: form.value.auth_method !== 'none' ? form.value.auth_method : null,
      auth_credentials: authCredentials,
      portal: form.value.portal || null,
      schedule_cron: form.value.enableSchedule ? form.value.schedule_cron : null,
      auto_embed: form.value.autoEmbed,
      hazard_type: form.value.hazard_type || null,
      region_country: form.value.region_country || null,
      spatial_coverage: form.value.spatial_coverage || null,
      impact_sector: form.value.impact_sector || null,
      keywords: form.value.keywords
        ? form.value.keywords.split(',').map(k => k.trim()).filter(Boolean)
        : null,
      custom_metadata: form.value.custom_metadata_raw
        ? Object.fromEntries(
            form.value.custom_metadata_raw.split(/[,\n]/)
              .map(p => p.split('=').map(s => s.trim()))
              .filter(p => p.length === 2 && p[0] && p[1])
          )
        : null,
    }
    Object.keys(sourceConfig).forEach(key => {
      if (sourceConfig[key] === null || sourceConfig[key] === undefined ||
          (Array.isArray(sourceConfig[key]) && sourceConfig[key].length === 0)) {
        delete sourceConfig[key]
      }
    })
    if (sourceConfig.time_range && Object.keys(sourceConfig.time_range).length === 0) delete sourceConfig.time_range

    const resp = await apiFetch('/sources', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sourceConfig)
    })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }))
      // FastAPI validation failures return `detail` as an array of
      // `{loc, msg, type}` objects — join them instead of letting JS render
      // "[object Object]" in the toast.
      const msg = Array.isArray(err.detail)
        ? err.detail.map(d => d.msg || JSON.stringify(d)).join('; ')
        : (err.detail || `HTTP ${resp.status}`)
      throw new Error(msg)
    }
    // The backend creates the source row synchronously and may have tried to
    // kick off auto-embed inline. If that failed, the response still returns
    // 201 but with `etl_error` or `error_message` populated — don't navigate
    // away silently, the user needs to know the ingest didn't start.
    const body = await resp.json().catch(() => ({}))
    const etlErr = body.etl_error || body.error_message
    if (etlErr) {
      toast.error(`Source created but ingestion did not start: ${etlErr}`)
    } else {
      toast.success(`Source "${body.source_id || sourceConfig.source_id}" created`)
    }
    router.push('/sources')
  } catch (e) {
    console.error('Error creating source:', e)
    toast.error(`Error creating source: ${e.message}`)
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
