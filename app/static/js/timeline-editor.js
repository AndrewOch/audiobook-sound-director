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
            timeInput: document.getElementById('timeline-time-input'),
            timeDisplay: document.getElementById('timeline-time-display'),
            zoomOut: document.getElementById('timeline-zoom-out'),
            zoomIn: document.getElementById('timeline-zoom-in'),
            zoomLevel: document.getElementById('timeline-zoom-level'),
            saveBtn: document.getElementById('timeline-save-btn'),
            openBtn: document.getElementById('timeline-open-btn'),
            selectionActions: document.getElementById('selection-actions'),
            selectionCount: document.getElementById('selection-count'),
            generateMusicBtn: document.getElementById('btn-generate-music'),
            generateFoliBtn: document.getElementById('btn-generate-foli'),
            foliChannelSelect: document.getElementById('foli-channel-select'),
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
                        <button class="btn-play" id="timeline-play-btn">
                            <span class="play-icon">▶</span>
                        </button>
                        <button class="btn-pause" id="timeline-pause-btn" style="display: none;">
                            <span class="pause-icon">❚❚</span>
                        </button>
                        <input type="number" class="time-input" id="timeline-time-input" value="0" step="0.1" min="0">
                        <span class="time-display" id="timeline-time-display">00:00 / 00:00</span>
                    </div>
                    <div class="toolbar-center">
                        <button class="btn-zoom-out" id="timeline-zoom-out">−</button>
                        <span class="zoom-level" id="timeline-zoom-level">100%</span>
                        <button class="btn-zoom-in" id="timeline-zoom-in">+</button>
                    </div>
                    <div class="toolbar-right">
                        <button class="btn-save" id="timeline-save-btn">💾 Сохранить</button>
                        <button class="btn-open" id="timeline-open-btn">📂 Открыть</button>
                    </div>
                </div>
                
                <!-- Selection Actions moved to bottom panel -->
                
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
                                <div class="playhead" id="playhead"></div>
                            </div>
                        </div>
                        
                        <!-- Generated Tracks -->
                        <div class="tracks-list" id="tracks-list"></div>
                    </div>
                </div>
            </div>

                <!-- Bottom Actions Panel -->
                <div class="bottom-actions" id="bottom-actions">
                    <!-- Selection section -->
                    <div class="selection-actions" id="selection-actions" style="display: none;">
                        <span class="selection-count" id="selection-count">0 сегментов выбрано</span>
                        <span class="selection-details" id="selection-details"></span>
                        <button class="btn-generate-music" id="btn-generate-music">🎵 Генерировать музыку</button>
                        <div class="foli-channel-select-wrap" style="display:flex;align-items:center;gap:6px;">
                            <label for="foli-channel-select">Канал:</label>
                            <select id="foli-channel-select">
                                <option value="ch1">ch1</option>
                                <option value="ch2">ch2</option>
                                <option value="ch3">ch3</option>
                            </select>
                        </div>
                        <button class="btn-generate-foli" id="btn-generate-foli">🔊 Генерировать фоли</button>
                        <button class="btn-clear-selection" id="btn-clear-selection">✕ Очистить</button>
                    </div>
                    <!-- Tracks mixer section -->
                    <div class="tracks-mixer" id="tracks-mixer"></div>
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
            saveBtn: document.getElementById('timeline-save-btn'),
            openBtn: document.getElementById('timeline-open-btn'),
            selectionActions: document.getElementById('selection-actions'),
            selectionCount: document.getElementById('selection-count'),
            generateMusicBtn: document.getElementById('btn-generate-music'),
            generateFoliBtn: document.getElementById('btn-generate-foli'),
            foliChannelSelect: document.getElementById('foli-channel-select'),
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
        
        // Selection actions
        if (this.elements.generateMusicBtn) this.elements.generateMusicBtn.addEventListener('click', () => this.generateForSelection('music'));
        if (this.elements.generateFoliBtn) this.elements.generateFoliBtn.addEventListener('click', () => this.generateForSelection('foli'));
        if (this.elements.clearSelectionBtn) this.elements.clearSelectionBtn.addEventListener('click', () => this.clearSelection());
        
        // Save/Open
        if (this.elements.saveBtn) this.elements.saveBtn.addEventListener('click', () => this.saveProject());
        if (this.elements.openBtn) this.elements.openBtn.addEventListener('click', () => this.openProjectDialog());
        
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
        
        // Click on timeline to seek
        if (this.elements.tracksArea) {
            this.elements.tracksArea.addEventListener('click', (e) => {
                if (e.target === this.elements.tracksArea || e.target.closest('.track-content')) {
                    const rect = this.elements.tracksArea.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const time = this.pixelsToTime(x);
                    this.seekTo(time);
                }
            });
        }
    }
    
    loadProject(projectData) {
        this.projectData = projectData;
        this.segments = projectData.segments || [];
        this.tracks = projectData.tracks || [];
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
        });
        this.audioElement.addEventListener('ended', () => {
            this.pause();
        });
        
        // Load waveform
        this.loadWaveform(audioUrl);
    }
    
    async loadWaveform(audioUrl) {
        const canvas = this.elements.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const duration = this.projectData?.duration || 0;
        const width = this.timeToPixels(duration);
        const height = this.options.trackHeight;
        
        canvas.width = width;
        canvas.height = height;
        
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
            
            // Render only visible viewport
            this.renderWaveformViewport();
            
            // Attach scroll/resize listeners for lazy rendering
            const onScroll = () => this.renderWaveformViewport();
            const onResize = () => this.renderWaveformViewport();
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
        this.renderEmotions();
        this.renderTracks();
        this.updatePlayhead();
        this.updateTimeDisplay();
        this.updateZoomDisplay();
    }
    
    renderTimeRuler() {
        const canvas = this.elements.rulerCanvas;
        const ctx = canvas.getContext('2d');
        const width = this.timeToPixels(this.projectData?.duration || 0);
        const height = this.options.headerHeight;
        
        canvas.width = width;
        canvas.height = height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        
        // Draw time markers
        const duration = this.projectData?.duration || 0;
        const interval = this.getTimeInterval();
        
        for (let time = 0; time <= duration; time += interval) {
            const x = this.timeToPixels(time);
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
            // Формируем строку фоли с учётом каналов
            let foliStr = '';
            if (segment.foli && typeof segment.foli === 'object') {
                const parts = [];
                if (segment.foli.ch1?.class) parts.push(`ch1: ${segment.foli.ch1.class}`);
                if (segment.foli.ch2?.class) parts.push(`ch2: ${segment.foli.ch2.class}`);
                if (segment.foli.ch3?.class) parts.push(`ch3: ${segment.foli.ch3.class}`);
                if (parts.length) foliStr = ` • Фоли: ${parts.join(', ')}`;
            } else if (foli) {
                foliStr = ` • Фоли: ${foli}${typeof foliConf === 'number' ? ` (${Math.round(foliConf * 100)}%)` : ''}`;
            }
            block.title = `Эмоция: ${emoStr}${foliStr}`;
            
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
                <div class="track-block" style="left: ${left}px; width: ${width}px;"></div>
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
        if (trackBlock) {
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
            this.audioElement.play();
            this.isPlaying = true;
            this.elements.playBtn.style.display = 'none';
            this.elements.pauseBtn.style.display = 'inline-block';
        }
        
        // Play all enabled tracks
        this.tracks.forEach(track => {
            if (track.enabled && track.audioElement) {
                track.audioElement.currentTime = this.playheadPosition - (track.start_time || 0);
                if (track.audioElement.currentTime >= 0 && track.audioElement.currentTime < track.audioElement.duration) {
                    track.audioElement.play();
                }
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
            if (track.audioElement) {
                const trackTime = this.playheadPosition - (track.start_time || 0);
                if (trackTime >= 0 && trackTime < track.audioElement.duration) {
                    track.audioElement.currentTime = trackTime;
                }
            }
        });
        
        this.updatePlayhead();
        this.updateTimeDisplay();
    }
    
    updatePlayhead() {
        const x = this.timeToPixels(this.playheadPosition);
        this.elements.playhead.style.left = `${x}px`;
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
                    if (seg.foli.ch1?.class) parts.push(`ch1=${seg.foli.ch1.class}`);
                    if (seg.foli.ch2?.class) parts.push(`ch2=${seg.foli.ch2.class}`);
                    if (seg.foli.ch3?.class) parts.push(`ch3=${seg.foli.ch3.class}`);
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
    }

    renderTracksMixer() {
        const mixer = this.elements.tracksMixer;
        if (!mixer) return;
        mixer.innerHTML = '';
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
                if (track.id === 'speech' && this.audioElement) {
                    // Управляем громкостью основного аудио речи в превью
                    this.audioElement.volume = vol;
                } else if (track.audioElement) {
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
            alert('Выберите сегменты для генерации');
            return;
        }
        
        let segmentIds = Array.from(this.selectedSegments);
        const jobId = this.projectData.job_id;
        let selectedChannel = undefined;
        if (type === 'foli') {
            selectedChannel = (this.elements.foliChannelSelect?.value || 'ch1');
            // Filter out segments where selected channel is Silence or missing
            const beforeCount = segmentIds.length;
            segmentIds = segmentIds.filter(id => {
                const seg = this.segments[id];
                const ch = seg?.foli?.[selectedChannel];
                const label = ch?.class || seg?.foli_class;
                return label && String(label).toLowerCase() !== 'silence';
            });
            if (segmentIds.length === 0) {
                alert('Нет сегментов для генерации: выбранный канал = Silence во всех выделенных сегментах');
                return;
            }
            if (segmentIds.length < beforeCount) {
                // Optional info
                console.info(`Пропущено ${beforeCount - segmentIds.length} сегмент(ов) с каналом Silence`);
            }
        }
        
        // Show loading state
        const generateBtn = type === 'music' ? this.elements.generateMusicBtn : this.elements.generateFoliBtn;
        const originalText = generateBtn.textContent;
        generateBtn.disabled = true;
        generateBtn.textContent = 'Генерация... 0%';
        
        try {
            // Create background task
            const response = await fetch(`/api/project/${jobId}/tasks/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    segment_ids: segmentIds,
                    type: type,
                    foli_channel: selectedChannel
                })
            });
            
            const data = await response.json();
            if (data.status === 'ok' && data.task_id) {
                const taskId = data.task_id;
                await this.pollGenerationTask(jobId, taskId, (progress, message) => {
                    generateBtn.textContent = `Генерация... ${progress}%`;
                });
                // After completion, reload project or add returned tracks
                try {
                    const taskResp = await fetch(`/api/project/${jobId}/tasks/${taskId}`);
                    const taskData = await taskResp.json();
                    if (taskData.state === 'completed' && Array.isArray(taskData.result_tracks)) {
                        taskData.result_tracks.forEach(track => this.addTrack(track));
                    } else {
                        // Fallback: reload full project
                        const pr = await fetch(`/api/project/${jobId}`);
                        const projectData = await pr.json();
                        this.tracks = projectData.tracks || [];
                        this.renderTracks();
                    }
                } catch (e) {
                    // ignore and fallback to alert
                }
                alert('Генерация завершена');
                this.clearSelection();
            } else {
                alert('Ошибка: ' + (data.detail || 'Неизвестная ошибка'));
            }
        } catch (error) {
            alert('Ошибка при генерации: ' + error.message);
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = originalText;
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
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TimelineEditor;
}

