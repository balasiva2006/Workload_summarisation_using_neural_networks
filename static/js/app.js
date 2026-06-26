/**
 * app.js — Dashboard logic for Feature 1
 * Fetches dashboard_data.json and renders stats, images, cluster cards
 */

(function () {
    'use strict';

    // ─── DOM refs ───
    const loadingScreen   = document.getElementById('loading-screen');
    const dashboard       = document.getElementById('dashboard');
    const btnToggle       = document.getElementById('btn-toggle-clusters');
    const clusterList     = document.getElementById('cluster-list');
    const modalOverlay    = document.getElementById('modal-overlay');
    const modalTitle      = document.getElementById('modal-title');
    const modalBody       = document.getElementById('modal-body');
    const modalClose      = document.getElementById('modal-close');

    let dashboardData = null;
    let clustersVisible = false;

    // ─── Init ───
    fetchDashboardData();

    async function fetchDashboardData() {
        try {
            const res = await fetch('/api/dashboard');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            dashboardData = await res.json();
            renderDashboard(dashboardData);
        } catch (err) {
            console.error('Failed to load dashboard data:', err);
            loadingScreen.innerHTML = `
                <div style="text-align:center;color:#f43f5e;">
                    <p style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">Failed to load data</p>
                    <p style="font-size:0.85rem;color:#94a3b8;">${err.message}</p>
                    <p style="font-size:0.8rem;color:#64748b;margin-top:1rem;">
                        Make sure you have run <code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;">python generate_data.py</code> first.
                    </p>
                </div>`;
        }
    }

    function renderDashboard(data) {
        // Stats
        document.getElementById('total-queries-hero').textContent = data.total_queries.toLocaleString();
        document.getElementById('stat-total-queries').textContent = data.total_queries.toLocaleString();
        document.getElementById('stat-optimal-k').textContent = data.optimal_k;
        document.getElementById('stat-compression').textContent = data.compression_ratio.toFixed(1) + '%';
        document.getElementById('stat-avg-cluster').textContent = data.avg_cluster_size.toFixed(0);

        // Images
        const ts = Date.now(); // cache-bust
        document.getElementById('elbow-img').src = '/' + data.images.elbow_curve + '?t=' + ts;
        document.getElementById('silhouette-img').src = '/' + data.images.silhouette_scores + '?t=' + ts;
        document.getElementById('pca-img').src = '/' + data.images.cluster_visualization + '?t=' + ts;

        // Show dashboard
        loadingScreen.style.display = 'none';
        dashboard.style.display = 'block';

        // Wire up cluster toggle
        btnToggle.addEventListener('click', toggleClusters);
    }

    function toggleClusters() {
        clustersVisible = !clustersVisible;
        if (clustersVisible) {
            renderClusterCards(dashboardData.clusters, clusterList);
            clusterList.style.display = 'grid';
            btnToggle.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                Hide Cluster Centers`;
        } else {
            clusterList.style.display = 'none';
            btnToggle.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                View All Cluster Centers`;
        }
    }

    // ─── Render cluster cards ───
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

    // ─── Modal ───
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

    // ─── Util ───
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Expose for upload page reuse
    window.WorkloadViz = { renderClusterCards, openModal, closeModal, escapeHtml };
})();
