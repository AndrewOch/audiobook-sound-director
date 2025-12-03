// Character counter
const textInput = document.getElementById('textInput');
const charCount = document.getElementById('charCount');

if (textInput && charCount) {
    textInput.addEventListener('input', () => {
        charCount.textContent = textInput.value.length;
    });
}

// Input type switching
const inputTypeRadios = document.querySelectorAll('input[name="inputType"]');
const textInputArea = document.getElementById('textInputArea');
const textFileArea = document.getElementById('textFileArea');
const audioFileArea = document.getElementById('audioFileArea');

inputTypeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        const value = e.target.value;

        // Hide all areas
        textInputArea.classList.add('hidden');
        textFileArea.classList.add('hidden');
        audioFileArea.classList.add('hidden');

        // Show selected area
        if (value === 'text') {
            textInputArea.classList.remove('hidden');
        } else if (value === 'text_file') {
            textFileArea.classList.remove('hidden');
        } else if (value === 'audio_file') {
            audioFileArea.classList.remove('hidden');
        }
    });
});

// Initialize visibility on load
window.addEventListener('DOMContentLoaded', () => {
    const selected = document.querySelector('input[name="inputType"]:checked');
    if (selected) {
        const event = new Event('change');
        selected.dispatchEvent(event);
    }
});

// File upload handlers
const textFileInput = document.getElementById('textFile');
const textFileDropZone = document.getElementById('textFileDropZone');
const textFileName = document.getElementById('textFileName');

if (textFileDropZone && textFileInput) {
    textFileDropZone.addEventListener('click', () => textFileInput.click());

    textFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            textFileName.textContent = `Выбрано: ${file.name}`;
            textFileName.classList.remove('hidden');
        }
    });
}

const audioFileInput = document.getElementById('audioFile');
const audioFileDropZone = document.getElementById('audioFileDropZone');
const audioFileName = document.getElementById('audioFileName');

if (audioFileDropZone && audioFileInput) {
    audioFileDropZone.addEventListener('click', () => audioFileInput.click());

    audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            audioFileName.textContent = `Выбрано: ${file.name}`;
            audioFileName.classList.remove('hidden');
        }
    });
}

// Form submission
const form = document.getElementById('audiobookForm');
const processingStatus = document.getElementById('processingStatus');
const trackEditor = document.getElementById('trackEditor');
const tracksContainer = document.getElementById('tracksContainer');
const playAllBtn = document.getElementById('playAllBtn');
const pauseAllBtn = document.getElementById('pauseAllBtn');
const downloadMixBtn = document.getElementById('downloadMixBtn');

let currentJobId = null;
let audioElements = []; // {id, el, enabled, volume}
let timelineEditor = null; // TimelineEditor instance

const speechResult = document.getElementById('speechResult');

async function synthesizeSpeechWithElevenLabs({ text, jobId } = {}) {
    if (!text) {
        throw new Error('Сначала введите текст');
    }

    const payload = { text };
    if (jobId) payload.job_id = jobId;

    const resp = await fetch('/api/speech/elevenlabs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (!resp.ok || data.status !== 'ok') {
        throw new Error(data.detail || data.error || 'Не удалось синтезировать речь');
    }
    return data;
}

// Open project modal from home button
const openProjectFromHomeBtn = document.getElementById('openProjectFromHome');
if (openProjectFromHomeBtn) {
    openProjectFromHomeBtn.addEventListener('click', () => openProjectModal());
}

// Modal controls and helpers
const projectModal = document.getElementById('projectModal');
const projectModalBackdrop = document.getElementById('projectModalBackdrop');
const projectModalClose = document.getElementById('projectModalClose');
const projectModalCancel = document.getElementById('projectModalCancel');
const projectListEl = document.getElementById('projectList');
const projectSearch = document.getElementById('projectSearch');

function showProjectModal() { if (projectModal) projectModal.classList.remove('hidden'); }
function hideProjectModal() { if (projectModal) projectModal.classList.add('hidden'); }

async function fetchProjects() {
    let serverProjects = [];
    try {
        const resp = await fetch('/api/projects');
        const data = await resp.json();
        serverProjects = data.projects || [];
    } catch {}
    const localProjects = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('project_')) {
            try {
                const p = JSON.parse(localStorage.getItem(key));
                localProjects.push({
                    job_id: p.job_id,
                    updated_at: p.updated_at,
                    created_at: p.created_at,
                    duration: p.duration,
                    source: 'local'
                });
            } catch {}
        }
    }
    const map = new Map();
    [...serverProjects, ...localProjects].forEach(p => { if (p && p.job_id) map.set(p.job_id, p); });
    return Array.from(map.values()).sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0));
}

function renderProjectList(projects) {
    if (!projectListEl) return;
    projectListEl.innerHTML = '';
    projects.forEach(p => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'w-full flex items-center justify-between px-3 py-3 hover:bg-gray-50';
        const left = document.createElement('div');
        left.className = 'flex flex-col text-left';
        const id = document.createElement('span');
        id.className = 'font-mono text-sm text-gray-800';
        id.textContent = p.job_id;
        const meta = document.createElement('span');
        meta.className = 'text-xs text-gray-500';
        meta.textContent = `Обновлён: ${new Date(p.updated_at || Date.now()).toLocaleString()} • Длительность: ${p.duration || 0}s`;
        left.appendChild(id);
        left.appendChild(meta);
        const right = document.createElement('span');
        right.className = 'material-icons-outlined text-gray-400';
        right.textContent = 'chevron_right';
        btn.appendChild(left);
        btn.appendChild(right);
        btn.addEventListener('click', async () => {
            hideProjectModal();
            await loadProjectToTimeline(p.job_id);
            const mainCard = document.getElementById('audiobookForm')?.parentElement;
            if (mainCard) mainCard.classList.add('hidden');
        });
        projectListEl.appendChild(btn);
    });
}

async function openProjectModal() {
    const projects = await fetchProjects();
    renderProjectList(projects);
    if (projectSearch) {
        projectSearch.value = '';
        projectSearch.oninput = (e) => {
            const q = e.target.value.toLowerCase();
            const filtered = projects.filter(p =>
                String(p.job_id).toLowerCase().includes(q) ||
                String(p.updated_at || '').toLowerCase().includes(q)
            );
            renderProjectList(filtered);
        };
    }
    showProjectModal();
}

if (projectModalBackdrop) projectModalBackdrop.addEventListener('click', hideProjectModal);
if (projectModalClose) projectModalClose.addEventListener('click', hideProjectModal);
if (projectModalCancel) projectModalCancel.addEventListener('click', hideProjectModal);

// Expose for timeline-editor reuse
window.__openProjectModal = openProjectModal;

if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        const inputType = document.querySelector('input[name="inputType"]:checked').value;

        if (inputType === 'text') {
            const text = textInput.value.trim();
            if (!text) {
                alert('Введите текст');
                return;
            }
            formData.append('text_input', text);
        } else if (inputType === 'text_file') {
            const file = textFileInput.files[0];
            if (!file) {
                alert('Выберите текстовый файл');
                return;
            }
            formData.append('text_file', file);
        } else if (inputType === 'audio_file') {
            const file = audioFileInput.files[0];
            if (!file) {
                alert('Выберите аудиофайл');
                return;
            }
            formData.append('audio_file', file);
        }

        formData.append('input_type', inputType);

        // Show processing status
        processingStatus.classList.remove('hidden');
        form.parentElement.classList.add('opacity-50', 'pointer-events-none');

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                alert('Ошибка: ' + (result.detail || 'Неизвестная ошибка'));
                processingStatus.classList.add('hidden');
                form.parentElement.classList.remove('opacity-50', 'pointer-events-none');
                return;
            }

            if (result.speech && speechResult) {
                const absoluteUrl = new URL(result.speech.audio_url, window.location.origin).toString();
                speechResult.innerHTML = `
                    <div class="mt-2 p-3 border border-primary/30 rounded-lg bg-indigo-50/30">
                        <p class="font-medium text-gray-800">Речь сгенерирована:</p>
                        <a href="${absoluteUrl}" target="_blank" class="text-primary underline">${result.speech.filename}</a>
                        <audio controls class="mt-2 w-full" src="${absoluteUrl}"></audio>
                    </div>
                `;
            }

            // Start polling job status until completed, then render track editor
            currentJobId = result.job_id;
            startStatusPolling(currentJobId);
        } catch (error) {
            console.error('Error:', error);
            alert('Произошла ошибка: ' + error.message);
            processingStatus.classList.add('hidden');
            form.parentElement.classList.remove('opacity-50', 'pointer-events-none');
        }
    });
}

async function loadProjectToTimeline(jobId) {
    try {
        currentJobId = jobId;
        
        // Load complete project data
        const response = await fetch(`/api/project/${jobId}`);
        if (!response.ok) {
            throw new Error('Failed to load project');
        }
        
        const projectData = await response.json();
        
        // Get timeline container
        const timelineContainer = document.getElementById('timelineEditorContainer');
        if (!timelineContainer) {
            console.error('Timeline container not found');
            throw new Error('Timeline container not found');
        }
        
        // Initialize timeline editor if not already initialized
        if (!timelineEditor) {
            timelineEditor = new TimelineEditor('timelineEditorContainer');
        }
        
        // Load project data
        timelineEditor.loadProject(projectData);
        
        // Show timeline editor
        timelineContainer.classList.remove('hidden');
        
        // Hide legacy editor
        const trackEditor = document.getElementById('trackEditor');
        if (trackEditor) {
            trackEditor.classList.add('hidden');
        }
    } catch (error) {
        try {
            console.error('Error loading project:', error, error && error.stack);
        } catch {}
        const msg = (error && (error.message || String(error))) || 'Неизвестная ошибка';
        alert('Ошибка загрузки проекта: ' + msg);
    }
}

function renderTrackEditor(tracks) {
    tracksContainer.innerHTML = '';
    audioElements = [];

    tracks.forEach(track => {
        const row = document.createElement('div');
        row.className = 'flex items-center justify-between border rounded-lg p-4';

        const left = document.createElement('div');
        left.className = 'flex items-center space-x-3';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = !!track.enabled;
        checkbox.className = 'h-5 w-5';
        const label = document.createElement('span');
        label.textContent = track.name;
        label.className = 'font-medium text-gray-800';
        left.appendChild(checkbox);
        left.appendChild(label);

        const right = document.createElement('div');
        right.className = 'flex items-center space-x-3 w-1/2';
        const range = document.createElement('input');
        range.type = 'range';
        range.min = '0';
        range.max = '1';
        range.step = '0.01';
        range.value = (typeof track.volume === 'number') ? track.volume : 1.0;
        range.className = 'w-full';

        right.appendChild(range);

        row.appendChild(left);
        row.appendChild(right);

        tracksContainer.appendChild(row);

        // Create audio element
        const audio = new Audio(track.url);
        audio.volume = parseFloat(range.value);
        audio.preload = 'auto';

        // Bind events
        checkbox.addEventListener('change', () => {
            if (!checkbox.checked) {
                audio.pause();
            }
        });
        range.addEventListener('input', () => {
            audio.volume = parseFloat(range.value);
        });

        audioElements.push({ id: track.id, el: audio, get enabled() { return checkbox.checked; }, get volume() { return parseFloat(range.value); } });
    });
}

function startStatusPolling(jobId) {
    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/status/${jobId}`);
            const data = await resp.json();
            if (!resp.ok) return;
            updateStepsFromStatus(data);

            if (data.status === 'completed') {
                clearInterval(interval);
                // Load project data and initialize timeline editor
                await loadProjectToTimeline(jobId);
                processingStatus.classList.add('hidden');
                form.parentElement.classList.add('hidden');
            }
            if (data.status === 'error') {
                clearInterval(interval);
                alert('Ошибка обработки: ' + (data.message || ''));
                processingStatus.classList.add('hidden');
                form.parentElement.classList.remove('opacity-50', 'pointer-events-none');
            }
        } catch (e) {
            // ignore transient errors
        }
    }, 2000);
}

function updateStepsFromStatus(statusData) {
    const steps = statusData.steps || {};
    setStepUI('step-ingest', steps.ingest?.status || 'pending', 'Входные данные получены', 'Ваши данные загружены');
    setStepUI('step-emotions', steps.emotion_analysis?.status || 'pending', 'Анализ эмоций', subtitleFor(steps.emotion_analysis));
    setStepUI('step-music', steps.music_generation?.status || 'pending', 'Генерация музыки', subtitleFor(steps.music_generation));
    setStepUI('step-foli', steps.foli_generation?.status || 'pending', 'Создание фоновых звуков', subtitleFor(steps.foli_generation));
    setStepUI('step-mixing', steps.mixing?.status || 'pending', 'Сведение финального аудио', subtitleFor(steps.mixing));
}

function subtitleFor(step) {
    const st = step?.status;
    if (st === 'running') return 'Выполняется...';
    if (st === 'completed') return 'Готово';
    if (st === 'skipped') return 'Пропущено';
    if (st === 'error') return (step?.detail ? ('Ошибка: ' + step.detail) : 'Ошибка');
    return 'Ожидание...';
}

function setStepUI(stepId, status, title, subtitle) {
    const el = document.getElementById(stepId);
    if (!el) return;
    const indicator = el.querySelector('div');
    const titleEl = el.querySelector('p.text-sm');
    const subEl = el.querySelector('p.text-xs');
    if (!indicator || !titleEl || !subEl) return;

    // Reset indicator
    indicator.className = 'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center';
    // Set styles based on status
    if (status === 'completed') {
        indicator.classList.add('bg-green-500');
        indicator.innerHTML = '<span class="material-icons-outlined text-white text-base">check</span>';
        titleEl.className = 'text-sm font-medium text-gray-900';
        subEl.className = 'text-xs text-gray-500';
    } else if (status === 'running') {
        indicator.classList.add('bg-primary');
        indicator.classList.add('animate-pulse');
        indicator.innerHTML = '<div class="w-3 h-3 bg-white rounded-full"></div>';
        titleEl.className = 'text-sm font-medium text-gray-900';
        subEl.className = 'text-xs text-gray-500';
    } else if (status === 'error') {
        indicator.classList.add('bg-red-500');
        indicator.innerHTML = '';
        titleEl.className = 'text-sm font-medium text-red-600';
        subEl.className = 'text-xs text-red-500';
    } else if (status === 'skipped') {
        indicator.classList.add('bg-gray-200');
        indicator.innerHTML = '';
        titleEl.className = 'text-sm font-medium text-gray-500';
        subEl.className = 'text-xs text-gray-400';
    } else {
        // pending
        indicator.classList.add('bg-gray-300');
        indicator.innerHTML = '';
        titleEl.className = 'text-sm font-medium text-gray-500';
        subEl.className = 'text-xs text-gray-400';
    }
    titleEl.textContent = title;
    subEl.textContent = subtitle || '';
}

if (playAllBtn) {
    playAllBtn.addEventListener('click', () => {
        const t = syncCurrentTime();
        audioElements.forEach(a => {
            if (a.enabled) {
                try { a.el.currentTime = t; } catch {}
                a.el.play();
            }
        });
    });
}

if (pauseAllBtn) {
    pauseAllBtn.addEventListener('click', () => {
        audioElements.forEach(a => a.el.pause());
    });
}

function syncCurrentTime() {
    // Use the first enabled track's currentTime as reference
    for (const a of audioElements) {
        if (a.enabled) return a.el.currentTime;
    }
    return 0;
}

if (downloadMixBtn) {
    downloadMixBtn.addEventListener('click', async () => {
        if (!currentJobId) return;
        const payload = {
            job_id: currentJobId,
            tracks: audioElements.map(a => ({ id: a.id, enabled: a.enabled, volume: a.volume }))
        };
        try {
            const resp = await fetch('/api/mix', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (!resp.ok || data.status !== 'ok') {
                alert('Ошибка при микшировании: ' + (data.detail || ''));
                return;
            }
            // Trigger download
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = 'mixed.wav';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (e) {
            alert('Ошибка: ' + e.message);
        }
    });
}
