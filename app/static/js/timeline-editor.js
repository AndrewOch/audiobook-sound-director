/**
 * Timeline Editor для аудиокниг
 * 
 * Полноценный редактор временной шкалы с визуализацией эмоций,
 * управлением дорожками и генерацией звуков.
 */

class TimelineEditor {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id "${containerId}" not found`);
        }
        
        this.options = {
            pixelsPerSecond: 50,
            trackHeight: 60,
            headerHeight: 40,
            ...options
        };
        
        this.projectData = null;
        this.audioElement = null;
        this.tracks = [];
        this.segments = [];
        this.selectedSegments = new Set();
        this.playheadPosition = 0.0;
        this.isPlaying = false;
        this.zoomLevel = 1.0;
        
        this.init();
    }
    
    init() {
        // Ensure editor HTML exists
        const hasEditor = this.container.querySelector('.timeline-editor');
        if (!hasEditor) {
            this.createHTML();
        }
        // (Re)collect element refs in case HTML already present
        this.elements = {
            playBtn: document.getElementById('timeline-play-btn'),
            pauseBtn: document.getElementById('timeline-pause-btn'),
            beginBtn: document.getElementById('timeline-begin'),
            rewindBtn: document.getElementById('timeline-rewind'),
            forwardBtn: document.getElementById('timeline-forward'),
            endBtn: document.getElementById('timeline-end'),
            timeInput: document.getElementById('timeline-time-input'),
            timeDisplay: document.getElementById('timeline-time-display'),
            zoomOut: document.getElementById('timeline-zoom-out'),
            zoomIn: document.getElementById('timeline-zoom-in'),
            zoomLevel: document.getElementById('timeline-zoom-level'),
            selectionActions: document.getElementById('selection-actions'),
            selectionCount: document.getElementById('selection-count'),
            generateMusicBtn: document.getElementById('btn-generate-music'),
            generateFoliBtn: document.getElementById('btn-generate-foli'),
            clearSelectionBtn: document.getElementById('btn-clear-selection'),
            timelineContainer: document.getElementById('timeline-container'),
            timeRuler: document.getElementById('time-ruler'),
            rulerCanvas: document.getElementById('ruler-canvas'),
            tracksArea: document.getElementById('tracks-area'),
            speechTrack: document.getElementById('speech-track'),
            waveformCanvas: document.getElementById('waveform-canvas'),
            emotionLayer: document.getElementById('emotion-layer'),
            playhead: document.getElementById('playhead'),
            tracksList: document.getElementById('tracks-list'),
            bottomActions: document.getElementById('bottom-actions'),
            tracksMixer: document.getElementById('tracks-mixer'),
            selectionDetails: document.getElementById('selection-details'),
            // toggles and panels
            selectionPanelToggle: document.getElementById('selection-panel-toggle'),
            selectionPanel: document.getElementById('selection-panel'),
            musicPanelToggle: document.getElementById('music-panel-toggle'),
            musicPanel: document.getElementById('music-panel'),
            foliPanelToggle: document.getElementById('foli-panel-toggle'),
            foliPanel: document.getElementById('foli-panel'),
            // controls
            musicLenInput: document.getElementById('music-length-seconds'),
            musicFadeLeftSelect: document.getElementById('music-transition-left'),
            musicFadeRightSelect: document.getElementById('music-transition-right'),
            statusLine: document.getElementById('status-line'),
            selectionInfoTable: document.getElementById('selection-info-table'),
        };
        this.setupEventListeners();
    }
    
    createHTML() {
        // Clear container and create HTML structure
        this.container.innerHTML = '';
        const editorDiv = document.createElement('div');
        editorDiv.className = 'timeline-editor-wrapper';
        editorDiv.innerHTML = `
            <div class="timeline-editor">
                <!-- Toolbar -->
                <div class="timeline-toolbar">
                    <div class="toolbar-left">
                        <button type="button" class="btn-play" id="timeline-play-btn" title="Воспроизвести">
                            <span class="material-icons-outlined">play_arrow</span>
                        </button>
                        <button type="button" class="btn-pause" id="timeline-pause-btn" style="display: none;" title="Пауза">
                            <span class="material-icons-outlined">pause</span>
                        </button>
                        <button type="button" class="btn-begin" id="timeline-begin" title="В начало">
                            <span class="material-icons-outlined">first_page</span>
                        </button>
                        <button type="button" class="btn-rewind" id="timeline-rewind" title="-10с">
                            <span class="material-icons-outlined">replay_10</span>
                        </button>
                        <button type="button" class="btn-forward" id="timeline-forward" title="+10с">
                            <span class="material-icons-outlined">forward_10</span>
                        </button>
                        <button type="button" class="btn-end" id="timeline-end" title="В конец">
                            <span class="material-icons-outlined">last_page</span>
                        </button>
                        <input type="number" class="time-input" id="timeline-time-input" value="0" step="0.1" min="0">
                        <span class="time-display" id="timeline-time-display">00:00 / 00:00</span>
                    </div>
                    <div class="toolbar-center">
                        <button class="btn-zoom-out" id="timeline-zoom-out">−</button>
                        <span class="zoom-level" id="timeline-zoom-level">100%</span>
                        <button class="btn-zoom-in" id="timeline-zoom-in">+</button>
                    </div>
                    <div class="toolbar-right"></div>
                </div>
                
                <!-- Timeline Container -->
                <div class="timeline-container" id="timeline-container">
                    <!-- Time Ruler -->
                    <div class="time-ruler" id="time-ruler">
                        <canvas id="ruler-canvas"></canvas>
                    </div>
                    
                    <!-- Tracks Area -->
                    <div class="tracks-area" id="tracks-area">
                        <!-- Speech Track (main audio) -->
                        <div class="track speech-track" id="speech-track">
                            <div class="track-label">Речь</div>
                            <div class="track-content">
                                <canvas id="waveform-canvas"></canvas>
                                <div class="emotion-layer" id="emotion-layer"></div>
                            </div>
                        </div>
                        <div class="playhead" id="playhead" title="Перетащите для перемотки"></div>
                        
                        <!-- Generated Tracks -->
                        <div class="tracks-list" id="tracks-list"></div>
                    </div>
                </div>
            </div>

                <!-- Bottom Actions Panel -->
                <div class="bottom-actions" id="bottom-actions">
                    <div class="ba-column">
                        <div class="panel">
                            <div class="panel-title with-toggle">
                                <button type="button" id="selection-panel-toggle" class="panel-toggle">Выделение</button>
                            </div>
                            <div class="panel-body" id="selection-panel">
                                <div class="selection-actions" id="selection-actions" style="display: none;">
                                    <span class="selection-count" id="selection-count">0 сегментов выбрано</span>
                                    <button class="btn-clear-selection" id="btn-clear-selection">✕ Очистить</button>
                                </div>
                                <table class="mini-table" id="selection-info-table"></table>
                                <div id="status-line" class="status-line"></div>
                            </div>
                        </div>
                        <div class="panel">
                            <div class="panel-title with-toggle">
                                <button type="button" id="music-panel-toggle" class="panel-toggle">Генерация музыки</button>
                            </div>
                            <div class="panel-body" id="music-panel" style="display:none;">
                                <div class="form-group">
                                    <label for="music-length-seconds" class="field-label">Длина (сек)</label>
                                    <input type="number" id="music-length-seconds" min="3" max="300" step="1" value="10" class="w-32 sm:w-40 md:w-48 bg-gray-800 text-gray-100 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500">
                                </div>
                                <div class="form-group">
                                    <label for="music-transition-left" class="field-label">Переход слева</label>
                                    <select id="music-transition-left" class="w-full sm:w-72 md:w-80 bg-gray-800 text-gray-100 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500">
                                        <option value="none">Без перехода</option>
                                        <option value="fade">Плавное затухание</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="music-transition-right" class="field-label">Переход справа</label>
                                    <select id="music-transition-right" class="w-full sm:w-72 md:w-80 bg-gray-800 text-gray-100 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500">
                                        <option value="none">Без перехода</option>
                                        <option value="fade">Плавное затухание</option>
                                    </select>
                                </div>
                                <button class="btn-generate-music" id="btn-generate-music">🎵 Сгенерировать музыку</button>
                            </div>
                        </div>
                        <div class="panel">
                            <div class="panel-title with-toggle">
                                <button type="button" id="foli-panel-toggle" class="panel-toggle">Генерация фоновых звуков</button>
                            </div>
                            <div class="panel-body" id="foli-panel" style="display:none;">
                                <div class="form-group">
                                    <label class="field-label">Каналы</label>
                                    <div class="flex items-center gap-4 flex-wrap">
                                        <label class="inline-flex items-center gap-2 text-gray-100"><input type="checkbox" name="foli-channel" value="ch1" checked class="accent-indigo-500"> <span>Акцентный</span></label>
                                        <label class="inline-flex items-center gap-2 text-gray-100"><input type="checkbox" name="foli-channel" value="ch2" class="accent-indigo-500"> <span>Фоновый</span></label>
                                        <label class="inline-flex items-center gap-2 text-gray-100"><input type="checkbox" name="foli-channel" value="ch3" class="accent-indigo-500"> <span>Вторичный</span></label>
                                    </div>
                                </div>
                                <button class="btn-generate-foli" id="btn-generate-foli">🔊 Сгенерировать фоли</button>
                            </div>
                        </div>
                    </div>
                    <div class="ba-column">
                        <div class="panel">
                            <div class="panel-title">Громкость дорожек</div>
                            <div class="panel-body">
                                <div class="tracks-mixer" id="tracks-mixer"></div>
                            </div>
                        </div>
                    </div>
                </div>
        `;
        this.container.appendChild(editorDiv);
        
        // Update container reference to the inner div
        const innerEditor = this.container.querySelector('.timeline-editor');
        if (!innerEditor) {
            throw new Error('Failed to create timeline editor HTML');
        }
        
        this.elements = {
            playBtn: document.getElementById('timeline-play-btn'),
            pauseBtn: document.getElementById('timeline-pause-btn'),
            timeInput: document.getElementById('timeline-time-input'),
            timeDisplay: document.getElementById('timeline-time-display'),
            zoomOut: document.getElementById('timeline-zoom-out'),
            zoomIn: document.getElementById('timeline-zoom-in'),
            zoomLevel: document.getElementById('timeline-zoom-level'),
            selectionActions: document.getElementById('selection-actions'),
            selectionCount: document.getElementById('selection-count'),
            generateMusicBtn: document.getElementById('btn-generate-music'),
            generateFoliBtn: document.getElementById('btn-generate-foli'),
            clearSelectionBtn: document.getElementById('btn-clear-selection'),
            timelineContainer: document.getElementById('timeline-container'),
            timeRuler: document.getElementById('time-ruler'),
            rulerCanvas: document.getElementById('ruler-canvas'),
            tracksArea: document.getElementById('tracks-area'),
            speechTrack: document.getElementById('speech-track'),
            waveformCanvas: document.getElementById('waveform-canvas'),
            emotionLayer: document.getElementById('emotion-layer'),
            playhead: document.getElementById('playhead'),
            tracksList: document.getElementById('tracks-list'),
            bottomActions: document.getElementById('bottom-actions'),
            tracksMixer: document.getElementById('tracks-mixer'),
            selectionDetails: document.getElementById('selection-details'),
            // new elements
            musicPanelToggle: document.getElementById('music-panel-toggle'),
            musicPanel: document.getElementById('music-panel'),
            foliPanelToggle: document.getElementById('foli-panel-toggle'),
            foliPanel: document.getElementById('foli-panel'),
            musicLenInput: document.getElementById('music-length-seconds'),
            musicFadeLeftSelect: document.getElementById('music-transition-left'),
            musicFadeRightSelect: document.getElementById('music-transition-right'),
            statusLine: document.getElementById('status-line'),
            selectionInfoTable: document.getElementById('selection-info-table'),
        };
        
        // Verify all elements are found
        for (const [key, element] of Object.entries(this.elements)) {
            if (!element) {
                console.warn(`Timeline element not found: ${key}`);
            }
        }
    }
    
    setupEventListeners() {
        // Playback controls
        if (this.elements.playBtn) this.elements.playBtn.addEventListener('click', () => this.play());
        if (this.elements.pauseBtn) this.elements.pauseBtn.addEventListener('click', () => this.pause());
        
        // Time navigation
        if (this.elements.timeInput) {
            this.elements.timeInput.addEventListener('change', (e) => {
                const time = parseFloat(e.target.value) || 0;
                this.seekTo(time);
            });
        }
        
        // Zoom controls
        if (this.elements.zoomIn) this.elements.zoomIn.addEventListener('click', () => this.zoomIn());
        if (this.elements.zoomOut) this.elements.zoomOut.addEventListener('click', () => this.zoomOut());
        // Transport controls
        if (this.elements.beginBtn) this.elements.beginBtn.addEventListener('click', () => this.seekTo(0));
        if (this.elements.endBtn) this.elements.endBtn.addEventListener('click', () => this.seekTo(this.projectData?.duration || 0));
        if (this.elements.rewindBtn) this.elements.rewindBtn.addEventListener('click', () => this.seekTo(Math.max(0, (this.playheadPosition - 10))));
        if (this.elements.forwardBtn) this.elements.forwardBtn.addEventListener('click', () => this.seekTo(Math.min((this.projectData?.duration || 0), (this.playheadPosition + 10))));
        
        // Panels toggles
        if (this.elements.selectionPanelToggle) this.elements.selectionPanelToggle.addEventListener('click', () => {
            const el = this.elements.selectionPanel; if (!el) return; el.style.display = (el.style.display === 'none' ? 'block' : 'none');
        });
        if (this.elements.musicPanelToggle) this.elements.musicPanelToggle.addEventListener('click', () => {
            const el = this.elements.musicPanel; if (!el) return; el.style.display = (el.style.display === 'none' ? 'block' : 'none');
        });
        if (this.elements.foliPanelToggle) this.elements.foliPanelToggle.addEventListener('click', () => {
            const el = this.elements.foliPanel; if (!el) return; el.style.display = (el.style.display === 'none' ? 'block' : 'none');
        });
        
        // Selection actions
        if (this.elements.generateMusicBtn) this.elements.generateMusicBtn.addEventListener('click', () => this.generateForSelection('music'));
        if (this.elements.generateFoliBtn) this.elements.generateFoliBtn.addEventListener('click', () => this.generateForSelection('foli'));
        if (this.elements.clearSelectionBtn) this.elements.clearSelectionBtn.addEventListener('click', () => this.clearSelection());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            if (e.code === 'Space') {
                e.preventDefault();
                if (this.isPlaying) this.pause();
                else this.play();
            } else if (e.code === 'ArrowLeft') {
                e.preventDefault();
                this.seekTo(Math.max(0, this.playheadPosition - 1));
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                this.seekTo(Math.min(this.projectData?.duration || 0, this.playheadPosition + 1));
            }
        });
        
        // Click on timeline to seek (consider label width)
        if (this.elements.tracksArea) {
            this.elements.tracksArea.addEventListener('click', (e) => {
                if (e.target === this.elements.tracksArea || e.target.closest('.track-content')) {
                    const rect = this.elements.tracksArea.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const labelW = this.getLabelWidth();
                    const time = this.pixelsToTime(Math.max(0, x - labelW));
                    this.seekTo(time);
                }
            });
        }
        
        // Drag playhead by head
        if (this.elements.playhead) {
            let dragging = false;
            const onDown = (e) => { dragging = true; e.preventDefault(); };
            const onMove = (e) => {
                if (!dragging) return;
                const rect = this.elements.tracksArea.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const time = this.pixelsToTime(Math.max(0, x - this.getLabelWidth()));
                this.seekTo(time);
            };
            const onUp = () => { dragging = false; };
            this.elements.playhead.addEventListener('mousedown', onDown);
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        }
        
        // Sync waveform on scroll/resize
        if (this.elements.timelineContainer) {
            this.elements.timelineContainer.addEventListener('scroll', () => this.renderWaveformViewport());
            window.addEventListener('resize', () => { this.updateScrollableWidths(); this.renderWaveformViewport(); });
        }
    }
    
    loadProject(projectData) {
        this.projectData = projectData;
        this.segments = projectData.segments || [];
        this.tracks = (projectData.tracks || []).filter(t => t.id !== 'speech');
        this.playheadPosition = projectData.playhead_position || 0.0;
        this.zoomLevel = projectData.zoom_level || 1.0;
        
        // Load audio
        if (projectData.audio_url) {
            this.loadAudio(projectData.audio_url);
        }
        
        // Render timeline
        this.render();
    }
    
    loadAudio(audioUrl) {
        this.audioElement = new Audio(audioUrl);
        this.audioElement.addEventListener('timeupdate', () => {
            this.playheadPosition = this.audioElement.currentTime;
            this.updatePlayhead();
            this.updateTimeDisplay();
            // While playing, start/stop generated tracks when playhead crosses their boundaries
            if (this.isPlaying) {
                this.updateTracksDuringPlayback();
            }
        });
        this.audioElement.addEventListener('ended', () => {
            this.pause();
        });
        // Setup WebAudio for speech gain control
        this.setupSpeechAudioGraph();
        
        // Load waveform
        this.loadWaveform(audioUrl);
    }
    
    setupSpeechAudioGraph() {
        try {
            if (this._audioGraphReady || !this.audioElement) return;
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            this._audioCtx = this._audioCtx || new AudioCtx();
            this._speechSource = this._speechSource || this._audioCtx.createMediaElementSource(this.audioElement);
            this._speechGainNode = this._speechGainNode || this._audioCtx.createGain();
            this._speechGainNode.gain.value = 1.0;
            this._speechSource.connect(this._speechGainNode);
            this._speechGainNode.connect(this._audioCtx.destination);
            this._audioGraphReady = true;
        } catch (e) {
            console.warn('Speech audio graph setup failed:', e);
        }
    }
    
    async loadWaveform(audioUrl) {
        const canvas = this.elements.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const duration = this.projectData?.duration || 0;
        const width = this.timeToPixels(duration);
        const height = this.options.trackHeight;
        
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = `${width}px`;
        
        // Show loading state (only viewport to reduce work)
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Загрузка waveform...', 8, height - 8);
        
        try {
            // Load audio file
            const response = await fetch(audioUrl);
            const arrayBuffer = await response.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Cache waveform data for lazy rendering
            const channelData = audioBuffer.getChannelData(0); // Mono
            const samplesPerPixel = Math.max(1, Math.floor(channelData.length / width));
            this._waveform = {
                channelData,
                samplesPerPixel,
                width,
                height
            };
            
            // Expand content widths to match duration
            this.updateScrollableWidths();
            
            // Render only visible viewport
            this.renderWaveformViewport();
            
            // Attach scroll/resize listeners for lazy rendering
            const onScroll = () => this.renderWaveformViewport();
            const onResize = () => { this.updateScrollableWidths(); this.renderWaveformViewport(); };
            if (!this._waveformListenersAttached) {
                this.elements.timelineContainer.addEventListener('scroll', onScroll);
                window.addEventListener('resize', onResize);
                this._waveformListenersAttached = true;
            }
        } catch (error) {
            console.error('Error loading waveform:', error);
            // Fallback: simple visualization
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#666';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Waveform недоступен', width / 2, height / 2);
        }
    }

    updateScrollableWidths() {
        const duration = this.projectData?.duration || 0;
        const w = this.timeToPixels(duration);
        if (this.elements.rulerCanvas) this.elements.rulerCanvas.style.width = `${w}px`;
        const speechContent = this.elements.speechTrack && this.elements.speechTrack.querySelector('.track-content');
        if (speechContent) speechContent.style.width = `${w}px`;
        if (this.elements.emotionLayer) this.elements.emotionLayer.style.width = `${w}px`;
        // Set width for each track-content
        if (this.elements.tracksList) {
            this.elements.tracksList.querySelectorAll('.track .track-content').forEach(el => {
                el.style.width = `${w}px`;
            });
        }
    }

    renderWaveformViewport() {
        const canvas = this.elements.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const wf = this._waveform;
        if (!wf) return;
        const { channelData, samplesPerPixel, width, height } = wf;
        const viewportWidth = this.elements.timelineContainer.clientWidth;
        const scrollLeft = this.elements.timelineContainer.scrollLeft;
        const startX = Math.max(0, Math.floor(scrollLeft));
        const endX = Math.min(width, Math.ceil(scrollLeft + viewportWidth + 50)); // small buffer
        
        // Clear only the viewport region
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(startX, 0, endX - startX, height);
        
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 1;
        const centerY = height / 2;
        
        for (let x = startX; x < endX; x++) {
            const start = Math.floor(x * samplesPerPixel);
            const end = Math.floor((x + 1) * samplesPerPixel);
            let min = 0;
            let max = 0;
            for (let i = start; i < end && i < channelData.length; i++) {
                const sample = channelData[i];
                if (sample < min) min = sample;
                if (sample > max) max = sample;
            }
            const amplitude = (max - min) * (height / 2) * 0.8;
            ctx.beginPath();
            ctx.moveTo(x + 0.5, centerY - amplitude);
            ctx.lineTo(x + 0.5, centerY + amplitude);
            ctx.stroke();
        }
    }
    
    render() {
        this.renderTimeRuler();
        this.updateScrollableWidths();
        this.renderEmotions();
        this.renderTracks();
        this.updatePlayhead();
        this.updateTimeDisplay();
        this.updateZoomDisplay();
    }
    
    renderTimeRuler() {
        const canvas = this.elements.rulerCanvas;
        const ctx = canvas.getContext('2d');
        const duration = this.projectData?.duration || 0;
        const contentW = this.timeToPixels(duration);
        const fullW = this.getLabelWidth() + contentW;
        const height = this.options.headerHeight;
        
        canvas.width = fullW;
        canvas.height = height;
        canvas.style.width = `${fullW}px`;
        
        ctx.clearRect(0, 0, fullW, height);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        
        // Draw time markers
        const interval = this.getTimeInterval();
        
        for (let time = 0; time <= duration; time += interval) {
            const x = this.getLabelWidth() + this.timeToPixels(time);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            
            // Time label
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.fillText(this.formatTime(time), x + 2, height - 5);
        }
    }
    
    renderEmotions() {
        const layer = this.elements.emotionLayer;
        layer.innerHTML = '';
        
        if (!this.segments) return;
        
        this.segments.forEach((segment, idx) => {
            const emotion = segment.emotion || 'neutral';
            const color = this.getEmotionColor(emotion);
            const left = this.timeToPixels(segment.start);
            const width = this.timeToPixels(segment.end - segment.start);
            const foli = segment.foli_class;
            const foliConf = segment.foli_confidence;
            
            const block = document.createElement('div');
            block.className = 'emotion-block';
            block.style.left = `${left}px`;
            block.style.width = `${width}px`;
            block.style.backgroundColor = color;
            block.style.opacity = 0.6;
            block.dataset.segmentId = idx;
            const emoStr = `${emotion} (${Math.round((segment.emotion_confidence || 0) * 100)}%)`;
            // Формируем строку фоли с учётом каналов (без Silence)
            let foliStr = '';
            if (segment.foli && typeof segment.foli === 'object') {
                const parts = [];
                if (segment.foli.ch1?.class && segment.foli.ch1.class.toLowerCase() !== 'silence') parts.push(`Акцентный: ${segment.foli.ch1.class}`);
                if (segment.foli.ch2?.class && segment.foli.ch2.class.toLowerCase() !== 'silence') parts.push(`Фоновый: ${segment.foli.ch2.class}`);
                if (segment.foli.ch3?.class && segment.foli.ch3.class.toLowerCase() !== 'silence') parts.push(`Вторичный: ${segment.foli.ch3.class}`);
                if (parts.length) foliStr = ` • Фоли: ${parts.join(', ')}`;
            } else if (foli) {
                foliStr = ` • Фоли: ${foli}${typeof foliConf === 'number' ? ` (${Math.round(foliConf * 100)}%)` : ''}`;
            }
            block.title = `Эмоция: ${emoStr}${foliStr}`;
            // Внутренняя подпись
            const label = document.createElement('div');
            label.className = 'emotion-label';
            label.textContent = `${emotion}${foliStr ? ' • ' + foliStr.replace(' • ', '') : ''}`;
            block.appendChild(label);
            
            block.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handleSegmentClick(e, idx);
            });
            
            layer.appendChild(block);
        });
    }
    
    renderTracks() {
        const tracksList = this.elements.tracksList;
        tracksList.innerHTML = '';
        
        this.tracks.forEach(track => {
            const trackEl = this.createTrackElement(track);
            tracksList.appendChild(trackEl);
        });
        // Set track-content width of all tracks to full timeline width
        this.updateScrollableWidths();
        // Render mixer after tracks are laid out
        this.renderTracksMixer();
    }
    
    createTrackElement(track) {
        const trackDiv = document.createElement('div');
        trackDiv.className = 'track generated-track';
        trackDiv.dataset.trackId = track.id;
        
        const left = this.timeToPixels(track.start_time || 0);
        // Try to get duration from track, or estimate from audio
        let duration = track.duration;
        if (!duration && track.url) {
            // Will be updated when audio loads
            duration = 10; // Default
        }
        const width = this.timeToPixels(duration);
        
        const trackName = track.name || track.id.replace(/_/g, ' ').replace(/segment/g, 'Сегмент');
        
        trackDiv.innerHTML = `
            <div class="track-label">
                <span>${trackName}</span>
                <button class="btn-delete-track" data-track-id="${track.id}">✕</button>
            </div>
            <div class="track-content">
                <div class="track-block${track.pending ? ' pending' : ''}" style="left: ${left}px; width: ${width}px;" title="${track.pending ? 'Генерация…' : ''}"></div>
            </div>
        `;
        
        // Load audio for this track
        if (track.url) {
            const audio = new Audio(track.url);
            audio.volume = track.volume || 0.5;
            track.audioElement = audio;
            
            audio.addEventListener('loadedmetadata', () => {
                track.duration = audio.duration;
                const newWidth = this.timeToPixels(audio.duration);
                const trackBlock = trackDiv.querySelector('.track-block');
                if (trackBlock) {
                    trackBlock.style.width = `${newWidth}px`;
                }
            });
        }
        
        // Setup drag
        const trackBlock = trackDiv.querySelector('.track-block');
        if (trackBlock && !track.pending) {
            this.setupTrackDrag(trackBlock, track);
        }
        
        // Delete button
        const deleteBtn = trackDiv.querySelector('.btn-delete-track');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                this.removeTrack(track.id);
            });
        }
        
        return trackDiv;
    }
    
    updatePendingTrackProgress(trackId, progress, type = 'music') {
        try {
            const trackEl = this.elements.tracksList?.querySelector(`.track[data-track-id="${trackId}"]`);
            if (!trackEl) return;
            const nameSpan = trackEl.querySelector('.track-label span');
            if (nameSpan) {
                const label = (type === 'foli') ? 'Фоли (генерация)…' : 'Музыка (генерация)…';
                nameSpan.textContent = `${label} ${Math.max(0, Math.min(100, Math.round(progress)))}%`;
            }
        } catch {}
    }
    
    setupTrackDrag(trackBlock, track) {
        let isDragging = false;
        let startX = 0;
        let startLeft = 0;
        
        trackBlock.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startLeft = parseFloat(trackBlock.style.left) || 0;
            trackBlock.style.cursor = 'grabbing';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const deltaX = e.clientX - startX;
            const newLeft = Math.max(0, startLeft + deltaX);
            trackBlock.style.left = `${newLeft}px`;
            track.start_time = this.pixelsToTime(newLeft);
        });
        
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                trackBlock.style.cursor = 'grab';
            }
        });
    }
    
    play() {
        if (this.audioElement) {
            this.audioElement.currentTime = this.playheadPosition;
            if (this._audioCtx && this._audioCtx.state === 'suspended') {
                try { this._audioCtx.resume(); } catch {}
            }
            this.audioElement.play();
            this.isPlaying = true;
            this.elements.playBtn.style.display = 'none';
            this.elements.pauseBtn.style.display = 'inline-block';
        }
        
        // Play all enabled tracks
        this.tracks.forEach(track => {
            if (!track.enabled || !track.audioElement) return;
            const offset = this.playheadPosition - (track.start_time || 0);
            const startAndPlay = () => {
                try {
                    track.audioElement.currentTime = Math.max(0, offset);
                    if (offset >= 0 && offset < (track.audioElement.duration || Infinity)) {
                        track.audioElement.play();
                    } else {
                        track.audioElement.pause();
                    }
                } catch {}
            };
            if (track.audioElement.readyState >= 1) {
                startAndPlay();
            } else {
                const onReady = () => {
                    track.audioElement.removeEventListener('loadedmetadata', onReady);
                    track.audioElement.removeEventListener('canplay', onReady);
                    startAndPlay();
                };
                track.audioElement.addEventListener('loadedmetadata', onReady, { once: true });
                track.audioElement.addEventListener('canplay', onReady, { once: true });
            }
        });
    }
    
    pause() {
        if (this.audioElement) {
            this.audioElement.pause();
            this.isPlaying = false;
            this.elements.playBtn.style.display = 'inline-block';
            this.elements.pauseBtn.style.display = 'none';
        }
        
        // Pause all tracks
        this.tracks.forEach(track => {
            if (track.audioElement) {
                track.audioElement.pause();
            }
        });
    }
    
    seekTo(time) {
        this.playheadPosition = Math.max(0, Math.min(time, this.projectData?.duration || 0));
        if (this.audioElement) {
            this.audioElement.currentTime = this.playheadPosition;
        }
        
        // Update all track positions
        this.tracks.forEach(track => {
            if (!track.audioElement) return;
            const desired = this.playheadPosition - (track.start_time || 0);
            const setTime = () => {
                try {
                    if (desired >= 0 && desired < (track.audioElement.duration || Infinity)) {
                        track.audioElement.currentTime = desired;
                    }
                } catch {}
            };
            if (track.audioElement.readyState >= 1) {
                setTime();
            } else {
                const onReady = () => {
                    track.audioElement.removeEventListener('loadedmetadata', onReady);
                    track.audioElement.removeEventListener('canplay', onReady);
                    setTime();
                };
                track.audioElement.addEventListener('loadedmetadata', onReady, { once: true });
                track.audioElement.addEventListener('canplay', onReady, { once: true });
            }
        });
        
        this.updatePlayhead();
        this.updateTimeDisplay();
    }
    
    updateTracksDuringPlayback() {
        // Ensure generated tracks begin playing when playhead enters their range,
        // and pause when playhead leaves it. Keep small drift correction.
        const DRIFT_SEC = 0.25;
        this.tracks.forEach(track => {
            const audio = track.audioElement;
            if (!audio) return;
            const offset = this.playheadPosition - (track.start_time || 0);
            const within = offset >= 0 && offset < (audio.duration || Infinity);
            if (this.isPlaying && within) {
                const startOrSync = () => {
                    try {
                        const desired = Math.max(0, offset);
                        const drift = Math.abs((audio.currentTime || 0) - desired);
                        if (audio.paused) {
                            audio.currentTime = desired;
                            audio.play().catch(() => {});
                        } else if (drift > DRIFT_SEC) {
                            audio.currentTime = desired;
                        }
                    } catch {}
                };
                if (audio.readyState >= 1) {
                    startOrSync();
                } else {
                    const onReady = () => {
                        audio.removeEventListener('loadedmetadata', onReady);
                        audio.removeEventListener('canplay', onReady);
                        startOrSync();
                    };
                    audio.addEventListener('loadedmetadata', onReady, { once: true });
                    audio.addEventListener('canplay', onReady, { once: true });
                }
            } else {
                if (!audio.paused) {
                    try { audio.pause(); } catch {}
                }
            }
        });
    }
    
    updatePlayhead() {
        const x = this.getLabelWidth() + this.timeToPixels(this.playheadPosition);
        this.elements.playhead.style.left = `${x}px`;
        this.elements.playhead.style.height = `${this.elements.tracksArea?.scrollHeight || 0}px`;
    }
    
    updateTimeDisplay() {
        const current = this.formatTime(this.playheadPosition);
        const total = this.formatTime(this.projectData?.duration || 0);
        this.elements.timeDisplay.textContent = `${current} / ${total}`;
        this.elements.timeInput.value = this.playheadPosition.toFixed(1);
    }
    
    updateZoomDisplay() {
        this.elements.zoomLevel.textContent = `${Math.round(this.zoomLevel * 100)}%`;
    }
    
    zoomIn() {
        this.zoomLevel = Math.min(5.0, this.zoomLevel * 1.2);
        this.options.pixelsPerSecond = 50 * this.zoomLevel;
        this.render();
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(0.1, this.zoomLevel / 1.2);
        this.options.pixelsPerSecond = 50 * this.zoomLevel;
        this.render();
    }
    
    toggleSegmentSelection(segmentId) {
        if (this.selectedSegments.has(segmentId)) {
            this.selectedSegments.delete(segmentId);
        } else {
            this.selectedSegments.add(segmentId);
        }
        this.updateSelectionUI();
    }
    
    handleSegmentClick(event, segmentId) {
        // Cmd/Ctrl+Click -> toggle add/remove
        if (event.metaKey || event.ctrlKey) {
            this.toggleSegmentSelection(segmentId);
            this._lastSelectedSegmentIndex = segmentId;
            return;
        }
        // Shift+Click -> select range from last anchor
        if (event.shiftKey) {
            const anchor = (typeof this._lastSelectedSegmentIndex === 'number') ? this._lastSelectedSegmentIndex : segmentId;
            const [start, end] = anchor <= segmentId ? [anchor, segmentId] : [segmentId, anchor];
            this.selectedSegments.clear();
            for (let i = start; i <= end; i++) this.selectedSegments.add(i);
            this.updateSelectionUI();
            return;
        }
        // Plain click -> single selection
        this.selectedSegments.clear();
        this.selectedSegments.add(segmentId);
        this._lastSelectedSegmentIndex = segmentId;
        this.updateSelectionUI();
    }
    
    clearSelection() {
        this.selectedSegments.clear();
        this.updateSelectionUI();
    }
    
    updateSelectionUI() {
        const count = this.selectedSegments.size;
        if (count > 0) {
            this.elements.selectionActions.style.display = 'flex';
            this.elements.selectionCount.textContent = `${count} сегмент${count > 1 ? 'ов' : ''} выбрано`;
            if (count === 1 && this.elements.selectionDetails) {
                const segId = Array.from(this.selectedSegments)[0];
                const seg = this.segments[segId];
                const emotion = seg?.emotion || '—';
                // Резюме по каналам фоли
                let foliSummary = '—';
                if (seg?.foli && typeof seg.foli === 'object') {
                    const parts = [];
                    if (seg.foli.ch1?.class) parts.push(`Акцентный=${seg.foli.ch1.class}`);
                    if (seg.foli.ch2?.class) parts.push(`Фоновый=${seg.foli.ch2.class}`);
                    if (seg.foli.ch3?.class) parts.push(`Вторичный=${seg.foli.ch3.class}`);
                    foliSummary = parts.length ? parts.join(', ') : '—';
                } else if (seg?.foli_class) {
                    foliSummary = seg.foli_class;
                }
                this.elements.selectionDetails.textContent = `Эмоция: ${emotion} | Фоли: ${foliSummary}`;
            } else if (this.elements.selectionDetails) {
                this.elements.selectionDetails.textContent = '';
            }
        } else {
            this.elements.selectionActions.style.display = 'none';
            if (this.elements.selectionDetails) {
                this.elements.selectionDetails.textContent = '';
            }
        }
        
        // Update visual selection
        this.elements.emotionLayer.querySelectorAll('.emotion-block').forEach((block, idx) => {
            if (this.selectedSegments.has(idx)) {
                block.classList.add('selected');
            } else {
                block.classList.remove('selected');
            }
        });
        // Update info table and defaults
        this.renderSelectionInfoTable();
    }

    renderTracksMixer() {
        const mixer = this.elements.tracksMixer;
        if (!mixer) return;
        mixer.innerHTML = '';
        // Add speech volume control (0..2)
        const speechItem = document.createElement('div');
        speechItem.className = 'mixer-item';
        const speechLabel = document.createElement('span');
        speechLabel.className = 'mixer-label';
        speechLabel.textContent = 'Речь';
        const speechSlider = document.createElement('input');
        speechSlider.type = 'range';
        speechSlider.className = 'volume-slider';
        speechSlider.min = '0';
        speechSlider.max = '2';
        speechSlider.step = '0.01';
        speechSlider.value = (this._speechGainNode ? this._speechGainNode.gain.value : 1.0).toString();
        speechSlider.addEventListener('input', (e) => {
            const vol = parseFloat(e.target.value);
            if (this._speechGainNode) {
                this._speechGainNode.gain.value = isFinite(vol) ? vol : 1.0;
            } else if (this.audioElement) {
                // Fallback: scale to 0..1
                this.audioElement.volume = Math.max(0, Math.min(1, vol / 2));
            }
        });
        speechItem.appendChild(speechLabel);
        speechItem.appendChild(speechSlider);
        mixer.appendChild(speechItem);
        this.tracks.forEach(track => {
            const item = document.createElement('div');
            item.className = 'mixer-item';
            const label = document.createElement('span');
            label.className = 'mixer-label';
            label.textContent = track.name || track.id;
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.className = 'volume-slider';
            slider.min = '0';
            slider.max = '1';
            slider.step = '0.01';
            slider.value = (track.volume ?? 0.5).toString();
            slider.addEventListener('input', (e) => {
                const vol = parseFloat(e.target.value);
                track.volume = vol;
                if (track.audioElement) {
                    track.audioElement.volume = vol;
                }
            });
            item.appendChild(label);
            item.appendChild(slider);
            mixer.appendChild(item);
        });
    }
    
    async generateForSelection(type) {
        if (this.selectedSegments.size === 0) {
            if (this.elements.selectionDetails) this.elements.selectionDetails.textContent = 'Выберите сегменты для генерации';
            return;
        }
        let segmentIds = Array.from(this.selectedSegments);
        const jobId = this.projectData.job_id;
        let selectedChannel = undefined;
        if (type === 'foli') {
            // Multi-channel checkboxes
            const selected = Array.from(document.querySelectorAll('input[name="foli-channel"]:checked')).map(el => el.value);
            if (selected.length === 0) {
                if (this.elements.selectionDetails) this.elements.selectionDetails.textContent = 'Выберите хотя бы один канал для генерации фоли';
                return;
            }
            if (selected.length === 1) selectedChannel = selected[0];
            if (selectedChannel) {
                const beforeCount = segmentIds.length;
                segmentIds = segmentIds.filter(id => {
                    const seg = this.segments[id];
                    const ch = seg?.foli?.[selectedChannel];
                    const label = ch?.class || seg?.foli_class;
                    return label && String(label).toLowerCase() !== 'silence';
                });
                if (segmentIds.length === 0) {
                    if (this.elements.selectionDetails) this.elements.selectionDetails.textContent = 'Нет сегментов для генерации: выбранный канал = Silence';
                    return;
                }
                if (this.elements.selectionDetails && beforeCount > segmentIds.length) {
                    this.elements.selectionDetails.textContent = `Пропущено ${beforeCount - segmentIds.length} сегмент(ов) с Silence`;
                }
            }
        }
        // Show loading state
        const generateBtn = type === 'music' ? this.elements.generateMusicBtn : this.elements.generateFoliBtn;
        const originalText = generateBtn.textContent; generateBtn.disabled = true; generateBtn.textContent = 'Генерация... 0%';
        // Pre-create pending music track
        let pendingId = null;
        if (type === 'music') {
            const ids = [...segmentIds].sort((a,b)=>a-b);
            const first = this.segments[ids[0]]; const last = this.segments[ids[ids.length-1]];
            const start = first?.start ?? 0; const end = last?.end ?? start;
            let lengthSec = parseInt(this.elements.musicLenInput?.value || '0', 10);
            if (!lengthSec || isNaN(lengthSec)) lengthSec = Math.max(3, Math.round(end - start));
            const transitionLeft = (this.elements.musicFadeLeftSelect?.value || 'none');
            const transitionRight = (this.elements.musicFadeRightSelect?.value || 'none');
            pendingId = `music_pending_${Date.now()}`;
            this.addTrack({
                id: pendingId,
                name: 'Музыка (генерация)…',
                type: 'music',
                start_time: start,
                duration: lengthSec,
                volume: 0.5,
                enabled: true,
                pending: true,
                transition_left: transitionLeft,
                transition_right: transitionRight
            });
        } else if (type === 'foli') {
            const ids = [...segmentIds].sort((a,b)=>a-b);
            const first = this.segments[ids[0]]; const last = this.segments[ids[ids.length-1]];
            const start = first?.start ?? 0; const end = last?.end ?? start;
            // Approximate duration across selection (backend generates per-segment; this is a visual placeholder)
            let lengthSec = Math.max(5, Math.round(end - start || (first?.end - first?.start) || 5));
            // Channel label
            const chLabel = selectedChannel ? ` (${selectedChannel})` : '';
            pendingId = `foli_pending_${Date.now()}`;
            this.addTrack({
                id: pendingId,
                name: `Фоли (генерация)…${chLabel}`,
                type: 'background',
                start_time: start,
                duration: lengthSec,
                volume: 0.6,
                enabled: true,
                pending: true
            });
        }
        try {
            const response = await fetch(`/api/project/${jobId}/tasks/generate`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    segment_ids: segmentIds,
                    type: type,
                    foli_channel: selectedChannel,
                    // optional music settings; backend may ignore if unsupported
                    music_length: (type === 'music') ? parseInt(this.elements.musicLenInput?.value || '0', 10) : undefined,
                    music_transition_left: (type === 'music') ? (this.elements.musicFadeLeftSelect?.value || 'none') : undefined,
                    music_transition_right: (type === 'music') ? (this.elements.musicFadeRightSelect?.value || 'none') : undefined
                })
            });
            const data = await response.json();
            if (data.status === 'ok' && data.task_id) {
                const taskId = data.task_id;
                await this.pollGenerationTask(jobId, taskId, (progress, message) => {
                    generateBtn.textContent = `Генерация... ${progress}%`;
                    if (pendingId) this.updatePendingTrackProgress(pendingId, progress, type);
                });
                try {
                    const taskResp = await fetch(`/api/project/${jobId}/tasks/${taskId}`);
                    const taskData = await taskResp.json();
                    if (pendingId) this.tracks = this.tracks.filter(t => t.id !== pendingId);
                    if (taskData.state === 'completed' && Array.isArray(taskData.result_tracks)) {
                        taskData.result_tracks.forEach(track => this.addTrack(track));
                    } else {
                        const pr = await fetch(`/api/project/${jobId}`); const projectData = await pr.json();
                        this.tracks = projectData.tracks || []; this.renderTracks();
                    }
                } catch (e) {}
                if (this.elements.selectionDetails) this.elements.selectionDetails.textContent = 'Генерация завершена';
                this.clearSelection();
            } else {
                if (this.elements.selectionDetails) this.elements.selectionDetails.textContent = 'Ошибка: ' + (data.detail || 'Неизвестная ошибка');
                if (pendingId) { this.tracks = this.tracks.filter(t => t.id !== pendingId); this.renderTracks(); }
            }
        } catch (error) {
            if (this.elements.selectionDetails) this.elements.selectionDetails.textContent = 'Ошибка при генерации: ' + error.message;
            if (pendingId) { this.tracks = this.tracks.filter(t => t.id !== pendingId); this.renderTracks(); }
        } finally {
            generateBtn.disabled = false; generateBtn.textContent = originalText;
        }
    }

    async pollGenerationTask(jobId, taskId, onProgress) {
        return new Promise((resolve) => {
            const interval = setInterval(async () => {
                try {
                    const resp = await fetch(`/api/project/${jobId}/tasks/${taskId}`);
                    if (!resp.ok) return;
                    const data = await resp.json();
                    if (typeof onProgress === 'function') {
                        onProgress(data.progress ?? 0, data.message || '');
                    }
                    if (data.state === 'completed' || data.state === 'error') {
                        clearInterval(interval);
                        resolve(data);
                    }
                } catch (e) {
                    // transient errors ignored
                }
            }, 1500);
        });
    }
    
    addTrack(track) {
        // Check if track already exists
        const existing = this.tracks.find(t => t.id === track.id);
        if (existing) {
            // Update existing track
            Object.assign(existing, track);
        } else {
            this.tracks.push(track);
        }
        this.renderTracks();
        this.renderTracksMixer();
    }
    
    removeTrack(trackId) {
        // Stop and remove audio element
        const track = this.tracks.find(t => t.id === trackId);
        if (track && track.audioElement) {
            track.audioElement.pause();
            track.audioElement = null;
        }
        this.tracks = this.tracks.filter(t => t.id !== trackId);
        this.renderTracks();
        this.renderTracksMixer();
    }
    
    saveProject() {
        if (!this.projectData) return;
        
        const projectState = {
            ...this.projectData,
            tracks: this.tracks,
            playhead_position: this.playheadPosition,
            zoom_level: this.zoomLevel,
            updated_at: new Date().toISOString()
        };
        
        // Save to localStorage
        const key = `project_${this.projectData.job_id}`;
        localStorage.setItem(key, JSON.stringify(projectState));
        
        // Also save to server
        fetch(`/api/project/${this.projectData.job_id}/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(projectState)
        });
        
        alert('Проект сохранен');
    }
    
    async openProjectDialog() {
        if (window.__openProjectModal) {
            window.__openProjectModal();
            return;
        }
        alert('Модальное окно выбора проекта недоступно');
    }
    
    // Utility methods
    timeToPixels(time) {
        return time * this.options.pixelsPerSecond;
    }
    
    pixelsToTime(pixels) {
        return pixels / this.options.pixelsPerSecond;
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    getTimeInterval() {
        // Adjust interval based on zoom level
        if (this.zoomLevel < 0.5) return 60; // 1 minute
        if (this.zoomLevel < 1.0) return 30; // 30 seconds
        if (this.zoomLevel < 2.0) return 10; // 10 seconds
        return 5; // 5 seconds
    }
    
    getEmotionColor(emotion) {
        const colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#DC143C',
            'fear': '#8B008B',
            'surprise': '#FF6347',
            'neutral': '#808080',
            'love': '#FF69B4',
            'excitement': '#FF4500',
            'gratitude': '#32CD32',
            'amusement': '#FFA500',
            'confusion': '#9370DB',
            'curiosity': '#00CED1',
            'disappointment': '#708090',
            'disgust': '#8B4513',
            'embarrassment': '#FF1493',
            'optimism': '#FFD700',
            'pride': '#FF8C00',
            'relief': '#90EE90',
            'remorse': '#B22222',
            'admiration': '#1E90FF',
            'annoyance': '#FF4500',
            'approval': '#00FF00',
            'caring': '#FFB6C1',
            'desire': '#FF1493',
            'disapproval': '#8B0000',
            'grief': '#2F4F4F',
            'nervousness': '#DA70D6',
            'realization': '#4682B4',
        };
        return colors[emotion] || '#808080';
    }

    getLabelWidth() {
        if (this._labelWidthCache && this._labelWidthCacheTime && (Date.now() - this._labelWidthCacheTime < 500)) return this._labelWidthCache;
        const lbl = this.elements.tracksArea?.querySelector('.track-label');
        const w = lbl ? (lbl.offsetWidth || 120) : 120;
        this._labelWidthCache = w; this._labelWidthCacheTime = Date.now();
        return w;
    }

    renderSelectionInfoTable() {
        const tbl = this.elements.selectionInfoTable; if (!tbl) return;
        const ids = Array.from(this.selectedSegments).sort((a,b)=>a-b);
        if (ids.length === 0) { tbl.innerHTML=''; return; }
        const first = this.segments[ids[0]]; const last = this.segments[ids[ids.length-1]];
        const start = first?.start ?? 0; const end = last?.end ?? start; const total = Math.max(0, (end - start));
        const rows = [];
        rows.push(`<tr><td>Начало</td><td>${start.toFixed(2)} с</td></tr>`);
        rows.push(`<tr><td>Конец</td><td>${end.toFixed(2)} с</td></tr>`);
        rows.push(`<tr><td>Длительность</td><td>${total.toFixed(2)} с</td></tr>`);
        // Helper to compute uniform value across selection
        const uniformOrDash = (getter) => {
            const values = ids.map(i => getter(this.segments[i]) || '').map(v => String(v));
            const nonEmpty = values.filter(v => v !== '');
            if (nonEmpty.length === 0) return ids.length === 1 ? '—' : '-';
            const firstVal = nonEmpty[0];
            const allSame = nonEmpty.every(v => v === firstVal);
            return allSame ? firstVal : '-';
        };
        // Emotion
        rows.push(`<tr><td>Эмоция</td><td>${uniformOrDash(seg => seg?.emotion)}</td></tr>`);
        // Foley channels
        rows.push(`<tr><td>Акцентный</td><td>${uniformOrDash(seg => seg?.foli?.ch1?.class)}</td></tr>`);
        rows.push(`<tr><td>Фоновый</td><td>${uniformOrDash(seg => seg?.foli?.ch2?.class)}</td></tr>`);
        rows.push(`<tr><td>Вторичный</td><td>${uniformOrDash(seg => seg?.foli?.ch3?.class)}</td></tr>`);
        tbl.innerHTML = rows.join(''); if (this.elements.musicLenInput) this.elements.musicLenInput.value = Math.max(3, Math.round(total || 0)) || 3;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TimelineEditor;
}

