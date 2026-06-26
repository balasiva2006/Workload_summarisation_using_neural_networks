/**
 * upload.js — File upload and clustering logic for Feature 2
 */

(function () {
    'use strict';

    const dropzone       = document.getElementById('dropzone');
    const fileInput      = document.getElementById('file-input');
    const fileInfo       = document.getElementById('file-info');
    const fileName       = document.getElementById('file-name');
    const fileRemove     = document.getElementById('file-remove');
    const kValueInput    = document.getElementById('k-value');
    const btnSubmit      = document.getElementById('btn-submit');
    const uploadForm     = document.getElementById('upload-form');
    const processing     = document.getElementById('processing');
    const progressBar    = document.getElementById('progress-bar');
    const resultsSection = document.getElementById('results-section');
    const clusterList    = document.getElementById('upload-cluster-list');
    const modalOverlay   = document.getElementById('modal-overlay');
    const modalTitle     = document.getElementById('modal-title');
    const modalBody      = document.getElementById('modal-body');
    const modalClose     = document.getElementById('modal-close');

    let selectedFile = null;

    // ─── Dropzone events ───
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        fileName.textContent = file.name + ' (' + formatBytes(file.size) + ')';
        fileInfo.style.display = 'flex';
        dropzone.style.display = 'none';
        btnSubmit.disabled = false;
    }

    fileRemove.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        dropzone.style.display = 'block';
        btnSubmit.disabled = true;
    });

    // ─── Form submit ───
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const k = parseInt(kValueInput.value, 10);
        if (isNaN(k) || k < 2 || k > 100) {
            alert('Please enter a valid K value between 2 and 100.');
            return;
        }

        // Show processing
        uploadForm.style.display = 'none';
        processing.style.display = 'block';
        resultsSection.style.display = 'none';

        // Animate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 8, 90);
            progressBar.style.width = progress + '%';
        }, 500);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('k', k);

            const res = await fetch('/api/cluster', {
                method: 'POST',
                body: formData,
            });

            clearInterval(progressInterval);
            progressBar.style.width = '100%';

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error || `HTTP ${res.status}`);
            }

            const data = await res.json();
            setTimeout(() => renderResults(data), 300);
        } catch (err) {
            clearInterval(progressInterval);
            processing.style.display = 'none';
            uploadForm.style.display = 'block';
            alert('Error: ' + err.message);
        }
    });

    function renderResults(data) {
        processing.style.display = 'none';

        // Stats
        document.getElementById('res-total-queries').textContent = data.total_queries.toLocaleString();
        document.getElementById('res-k').textContent = data.k;
        document.getElementById('res-compression').textContent = data.compression_ratio.toFixed(1) + '%';
        document.getElementById('res-avg-cluster').textContent = data.avg_cluster_size.toFixed(0);

        // Cluster cards
        renderClusterCards(data.clusters, clusterList);

        resultsSection.style.display = 'block';
    }

    // ─── Cluster cards + modal (shared logic) ───
    function renderClusterCards(clusters, container) {
        container.innerHTML = '';
        clusters.forEach(c => {
            const card = document.createElement('div');
            card.className = 'cluster-card';
            card.innerHTML = `
                <div class="cluster-card-header">
                    <span class="cluster-id">Cluster ${c.cluster_id}</span>
                    <span class="cluster-count">${c.query_count.toLocaleString()} queries</span>
                </div>
                <div class="cluster-center-query">${escapeHtml(c.center_query)}</div>
                <div class="cluster-card-footer">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                    View sample queries
                </div>`;
            card.addEventListener('click', () => openModal(c));
            container.appendChild(card);
        });
    }

    function openModal(cluster) {
        modalTitle.textContent = `Cluster ${cluster.cluster_id} — ${cluster.query_count.toLocaleString()} Queries`;
        modalBody.innerHTML = '';

        cluster.sample_queries.forEach((q, i) => {
            const item = document.createElement('div');
            item.className = 'query-item';
            item.innerHTML = `<span class="query-index">#${i + 1}</span>${escapeHtml(q)}`;
            modalBody.appendChild(item);
        });

        modalOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        modalOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }

    modalClose.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
})();
