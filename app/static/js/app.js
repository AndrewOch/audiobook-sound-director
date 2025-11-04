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
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Произошла ошибка: ' + error.message);
        } finally {
            // Reset UI after demo
            setTimeout(() => {
                processingStatus.classList.add('hidden');
                form.parentElement.classList.remove('opacity-50', 'pointer-events-none');
            }, 3000);
        }
    });
}
