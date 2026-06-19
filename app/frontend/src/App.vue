<script setup>
import { computed, onMounted, ref, watch } from "vue";

const apiBase = import.meta.env.VITE_API_BASE_URL || "";

const models = ref([]);
const selectedModel = ref("beto");
const searchQuery = ref("");
const articles = ref([]);
const selectedIndex = ref(0);
const loadingNews = ref(false);
const classifying = ref(false);
const evaluating = ref(false);
const errorMessage = ref("");
const result = ref(null);
const evaluation = ref(null);

const selectedArticle = computed(() => articles.value[selectedIndex.value] ?? null);

watch(selectedIndex, () => {
  result.value = null;
  evaluation.value = null;
  errorMessage.value = "";
});

async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = "Se produjo un error inesperado.";
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }

  return response.json();
}

async function loadModels() {
  const payload = await request("/api/models");
  models.value = payload;
  if (payload.length && !selectedModel.value) {
    selectedModel.value = payload[0].id;
  }
}

async function loadNews() {
  loadingNews.value = true;
  errorMessage.value = "";
  result.value = null;

  try {
    const query = searchQuery.value.trim();
    const suffix = query ? `?query=${encodeURIComponent(query)}` : "";
    const payload = await request(`/api/news${suffix}`);
    articles.value = payload.articles || [];
    selectedIndex.value = 0;
    if (!articles.value.length) {
      errorMessage.value = "No se encontraron noticias para esa consulta.";
    }
  } catch (error) {
    errorMessage.value = error.message;
    articles.value = [];
  } finally {
    loadingNews.value = false;
  }
}

async function classifySelected() {
  if (!selectedArticle.value) {
    errorMessage.value = "Selecciona primero una noticia.";
    return;
  }

  classifying.value = true;
  evaluation.value = null;
  errorMessage.value = "";

  try {
    result.value = await request("/api/classify", {
      method: "POST",
      body: JSON.stringify({
        model_id: selectedModel.value,
        article: selectedArticle.value,
      }),
    });
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    classifying.value = false;
  }
}

async function evaluateJustification() {
  if (!result.value || !selectedArticle.value) return;
  evaluating.value = true;
  errorMessage.value = "";
  try {
    evaluation.value = await request("/api/evaluate", {
      method: "POST",
      body: JSON.stringify({
        article: selectedArticle.value,
        prediction_label: result.value.prediction_label,
        justification: result.value.justification,
      }),
    });
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    evaluating.value = false;
  }
}

function probabilityText(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function scoreColor(val) {
  if (val >= 4) return "#69e3b7";
  if (val >= 3) return "#fbbf24";
  return "#ff7c96";
}

onMounted(async () => {
  try {
    await loadModels();
    await loadNews();
  } catch (error) {
    errorMessage.value = error.message;
  }
});
</script>

<template>
  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Detección de noticias falsas</p>
        <h1>DetectIA</h1>
        <p class="lede">
          Recupera una noticia en español, clasifícala con BETO y genera una
          justificación corta con un LLM.
        </p>
      </div>
    </section>

    <section class="panel controls">
      <div class="field grow">
        <label for="query">Buscar noticia</label>
        <input
          id="query"
          v-model="searchQuery"
          type="text"
          placeholder="Ej. elecciones, salud pública, economía..."
          @keyup.enter="loadNews"
        />
      </div>

      <div class="field">
        <label for="model">Modelo</label>
        <select id="model" v-model="selectedModel">
          <option v-for="model in models" :key="model.id" :value="model.id">
            {{ model.name }}
          </option>
        </select>
      </div>

      <div class="actions">
        <button class="button ghost" :disabled="loadingNews" @click="loadNews">
          {{ loadingNews ? "Cargando..." : "Obtener noticias" }}
        </button>
        <button class="button" :disabled="classifying || !selectedArticle" @click="classifySelected">
          {{ classifying ? "Clasificando..." : "Clasificar noticia" }}
        </button>
      </div>
    </section>

    <p v-if="errorMessage" class="status error">{{ errorMessage }}</p>

    <section class="layout">
      <aside class="panel list-panel">
        <div class="panel-head">
          <h2>Noticias disponibles</h2>
          <span>{{ articles.length }}</span>
        </div>

        <div v-if="!articles.length" class="empty">
          Aún no hay noticias cargadas.
        </div>

        <button
          v-for="(article, index) in articles"
          :key="`${article.url || article.title}-${index}`"
          class="article-item"
          :class="{ active: index === selectedIndex }"
          @click="selectedIndex = index"
        >
          <span class="article-source">{{ article.source.name || "Fuente desconocida" }}</span>
          <strong>{{ article.title }}</strong>
          <small>{{ article.published_at ? article.published_at.slice(0, 10) : "Sin fecha" }}</small>
        </button>
      </aside>

      <section class="panel detail-panel">
        <div v-if="selectedArticle" class="article-detail">
          <div class="panel-head">
            <h2>Noticia seleccionada</h2>
            <a v-if="selectedArticle.url" :href="selectedArticle.url" target="_blank" rel="noreferrer">
              Abrir fuente
            </a>
          </div>

          <h3>{{ selectedArticle.title }}</h3>
          <p class="meta">
            {{ selectedArticle.source.name || "Fuente desconocida" }}
            <span>·</span>
            {{ selectedArticle.published_at ? selectedArticle.published_at.slice(0, 10) : "Sin fecha" }}
          </p>
          <p class="body">{{ selectedArticle.body }}</p>
        </div>

        <div v-else class="empty">
          Selecciona una noticia para ver el detalle.
        </div>

        <div v-if="result" class="result-card">
          <div class="result-top">
            <div>
              <span class="result-label">Predicción</span>
              <h3 :class="['prediction', result.prediction === 'fake' ? 'fake' : 'real']">
                {{ result.prediction_label }}
              </h3>
            </div>
            <div class="probabilities">
              <div>
                <span>Fake</span>
                <strong>{{ probabilityText(result.fake_probability) }}</strong>
              </div>
              <div>
                <span>Real</span>
                <strong>{{ probabilityText(result.real_probability) }}</strong>
              </div>
            </div>
          </div>

          <div class="result-block">
            <span class="result-label">Justificación</span>
            <p>{{ result.justification }}</p>
          </div>

          <details class="result-block">
            <summary>Texto preprocesado enviado a BETO</summary>
            <p>{{ result.processed_text }}</p>
          </details>

          <div class="result-block">
            <button
              class="button ghost"
              :disabled="evaluating"
              @click="evaluateJustification"
            >
              {{ evaluating ? "Evaluando..." : "Evaluar justificación (G-Eval)" }}
            </button>
          </div>

          <div v-if="evaluation" class="result-block">
            <span class="result-label">Evaluación G-Eval — LLM como juez</span>
            <div class="geval-grid">
              <div v-for="dim in [
                { key: 'relevancia',  label: 'Relevancia',  desc: 'Basada en el artículo' },
                { key: 'fidelidad',   label: 'Fidelidad',   desc: 'Fiel al texto original' },
                { key: 'coherencia',  label: 'Coherencia',  desc: 'Clara y bien argumentada' },
                { key: 'alineacion',  label: 'Alineación',  desc: 'Encaja con la predicción' },
              ]" :key="dim.key" class="geval-item">
                <span class="geval-label">{{ dim.label }}</span>
                <span class="geval-desc">{{ dim.desc }}</span>
                <span class="geval-score" :style="{ color: scoreColor(evaluation[dim.key]) }">
                  {{ evaluation[dim.key] }}<small>/5</small>
                </span>
              </div>
            </div>
            <div class="geval-media">
              <span>Puntuación media</span>
              <strong :style="{ color: scoreColor(evaluation.media) }">
                {{ evaluation.media.toFixed(2) }} / 5
              </strong>
            </div>
            <details v-if="evaluation.razonamiento" class="geval-reasoning">
              <summary>Razonamiento del juez (Chain-of-Thought)</summary>
              <p>{{ evaluation.razonamiento }}</p>
            </details>
          </div>
        </div>
      </section>
    </section>
  </main>
</template>
