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

if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        const inputType = document.querySelector('input[name="inputType"]:checked').value;

        formData.append('input_type', inputType);

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
                alert('Ошибка: ' + result.detail);
                processingStatus.classList.add('hidden');
                form.parentElement.classList.remove('opacity-50', 'pointer-events-none');
                return;
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
                // Load tracks
                const tr = await fetch(`/api/tracks/${jobId}`);
                const tracks = await tr.json();
                renderTrackEditor(tracks);
                processingStatus.classList.add('hidden');
                form.parentElement.classList.add('hidden');
                trackEditor.classList.remove('hidden');
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
