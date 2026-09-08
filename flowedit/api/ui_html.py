"""
FlowEdit Web Interface HTML Template.
"""

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowEdit — Pronunciation Adaptation & Speech Synthesis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23DC2626' stroke-width='2.5'><polygon points='11 5 6 9 2 9 2 15 6 15 11 19 11 5'/><path d='M19.07 4.93a10 10 0 0 1 0 14.14'/><path d='M15.54 8.46a5 5 0 0 1 0 7.07'/></svg>">
    
    <style>
        :root {
            --primary: #DC2626;
            --primary-dark: #B91C1C;
            --primary-light: #EF4444;
            --primary-bg: #FEF2F2;
            --primary-glow: rgba(220, 38, 38, 0.25);
            --primary-border: rgba(220, 38, 38, 0.18);
            
            --bg-page: #F8FAFC;
            --bg-card: #FFFFFF;
            --bg-input: #FFFFFF;
            --bg-hover: #F1F5F9;
            
            --text-main: #0F172A;
            --text-body: #334155;
            --text-muted: #64748B;
            --text-light: #94A3B8;
            
            --border-subtle: #E2E8F0;
            --border-strong: #CBD5E1;
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-full: 9999px;
            
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
            --shadow-lg: 0 10px 30px -5px rgba(220, 38, 38, 0.10);
            --shadow-red: 0 8px 24px -4px rgba(220, 38, 38, 0.35);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-body);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* ── Header ─────────────────────────────────────────────── */
        header {
            height: 70px;
            background: #FFFFFF;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            z-index: 50;
            flex-shrink: 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            box-shadow: var(--shadow-red);
        }

        .brand-logo svg {
            width: 24px;
            height: 24px;
        }

        .brand-info h1 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .brand-info .tagline {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: -0.01em;
        }

        .header-center {
            display: flex;
            align-items: center;
            background: #F1F5F9;
            padding: 4px;
            border-radius: var(--radius-full);
            gap: 4px;
        }

        .nav-tab {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 18px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            border: none;
            background: transparent;
            border-radius: var(--radius-full);
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }

        .nav-tab:hover {
            color: var(--text-main);
        }

        .nav-tab.active {
            background: #FFFFFF;
            color: var(--primary);
            box-shadow: var(--shadow-sm);
        }

        .nav-tab svg {
            width: 16px;
            height: 16px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--primary-bg);
            border: 1px solid var(--primary-border);
            border-radius: var(--radius-full);
            font-size: 12px;
            font-weight: 600;
            color: var(--primary);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 8px #22C55E;
        }

        /* ── Main Container ─────────────────────────────────────── */
        .app-container {
            display: flex;
            flex: 1;
            overflow: hidden;
            background: var(--bg-page);
        }

        .tab-content {
            display: none;
            width: 100%;
            height: 100%;
        }

        .tab-content.active {
            display: flex;
        }

        /* ── Tab 1: Synthesis Studio Layout ────────────────────── */
        .studio-layout {
            display: flex;
            width: 100%;
            height: 100%;
        }

        aside.sidebar {
            width: 320px;
            background: var(--bg-card);
            border-right: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            padding: 1.5rem 1.25rem;
            gap: 1.5rem;
            overflow-y: auto;
            flex-shrink: 0;
        }

        .sidebar-section-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Model selector cards */
        .model-picker {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .model-option {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: #FFFFFF;
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .model-option:hover {
            border-color: var(--border-strong);
            background: #F8FAFC;
        }

        .model-option.selected {
            border-color: var(--primary);
            background: var(--primary-bg);
        }

        .model-option-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .custom-checkbox {
            width: 18px;
            height: 18px;
            border-radius: 5px;
            border: 2px solid var(--border-strong);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
        }

        .model-option.selected .custom-checkbox {
            background: var(--primary);
            border-color: var(--primary);
        }

        .custom-checkbox svg {
            width: 12px;
            height: 12px;
            color: #FFFFFF;
            display: none;
        }

        .model-option.selected .custom-checkbox svg {
            display: block;
        }

        .model-name {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
        }

        .model-pill {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: var(--radius-full);
            background: var(--primary);
            color: #FFFFFF;
            text-transform: uppercase;
        }

        .model-pill.base {
            background: #64748B;
        }

        /* Voice Selector */
        .voice-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .voice-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.625rem 0.875rem;
            background: #FFFFFF;
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .voice-card:hover {
            border-color: var(--border-strong);
            background: #F8FAFC;
        }

        .voice-card.selected {
            border-color: var(--primary);
            background: var(--primary-bg);
        }

        .voice-card-left {
            display: flex;
            align-items: center;
            gap: 0.625rem;
        }

        .voice-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #E2E8F0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 12px;
            color: var(--text-main);
        }

        .voice-card.selected .voice-avatar {
            background: var(--primary);
            color: #FFFFFF;
        }

        .voice-info {
            display: flex;
            flex-direction: column;
        }

        .voice-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
        }

        .voice-desc {
            font-size: 11px;
            color: var(--text-muted);
        }

        .voice-play-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            border-radius: 50%;
            transition: all 0.15s ease;
        }

        .voice-play-btn:hover {
            color: var(--primary);
            background: rgba(220,38,38,0.1);
        }

        /* Voice upload trigger */
        .upload-voice-box {
            border: 1.5px dashed var(--border-strong);
            border-radius: var(--radius-md);
            padding: 0.875rem;
            text-align: center;
            cursor: pointer;
            background: #FAFAFA;
            transition: all 0.2s ease;
        }

        .upload-voice-box:hover {
            border-color: var(--primary);
            background: var(--primary-bg);
        }

        .upload-voice-box span {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
        }

        /* Main Workspace */
        main.workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 2rem;
            gap: 1.5rem;
            overflow-y: auto;
        }

        .prompt-card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-md);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .prompt-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .prompt-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 16px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .sample-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }

        .sample-chip {
            font-size: 12px;
            padding: 4px 10px;
            background: #F1F5F9;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-full);
            color: var(--text-body);
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .sample-chip:hover {
            background: var(--primary-bg);
            border-color: var(--primary-border);
            color: var(--primary);
        }

        .text-input-area {
            width: 100%;
            min-height: 120px;
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1rem;
            font-family: inherit;
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-main);
            resize: vertical;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .text-input-area:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.12);
        }

        .prompt-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .char-counter {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .action-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 14px;
            font-weight: 700;
            border-radius: var(--radius-md);
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: #FFFFFF;
            box-shadow: var(--shadow-red);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 10px 28px -4px rgba(220, 38, 38, 0.45);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: #FFFFFF;
            border: 1.5px solid var(--border-subtle);
            color: var(--text-body);
        }

        .btn-secondary:hover {
            background: #F8FAFC;
            border-color: var(--border-strong);
        }

        /* Audio Outputs Grid */
        .outputs-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 1.25rem;
        }

        .output-card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1.5px solid var(--border-subtle);
            box-shadow: var(--shadow-md);
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            position: relative;
            overflow: hidden;
            transition: all 0.2s ease;
        }

        .output-card.finetuned {
            border-color: var(--primary-border);
        }

        .output-card.finetuned::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        }

        .output-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .output-tag {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            font-weight: 700;
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-main);
        }

        .output-tag.finetuned {
            color: var(--primary);
        }

        .metrics-row {
            display: flex;
            gap: 1rem;
            padding: 0.5rem 0.75rem;
            background: #F8FAFC;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
        }

        .metric-item {
            display: flex;
            flex-direction: column;
        }

        .metric-label {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
        }

        .player-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        audio {
            width: 100%;
            height: 40px;
            border-radius: var(--radius-sm);
            outline: none;
        }

        .dl-button {
            width: 40px;
            height: 40px;
            border-radius: var(--radius-sm);
            background: #F1F5F9;
            border: 1px solid var(--border-subtle);
            color: var(--text-body);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
            text-decoration: none;
            flex-shrink: 0;
        }

        .dl-button:hover {
            background: var(--primary-bg);
            color: var(--primary);
            border-color: var(--primary-border);
        }

        /* ── Tab 2: FlowEdit Pronunciation Learner ─────────────── */
        .learner-container {
            width: 100%;
            height: 100%;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            overflow-y: auto;
        }

        .learner-card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border-subtle);
            box-shadow: var(--shadow-md);
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .stage-indicator {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 1rem;
        }

        .stage-step {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem;
            background: #F8FAFC;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            transition: all 0.2s ease;
        }

        .stage-step.active {
            background: var(--primary-bg);
            border-color: var(--primary-border);
        }

        .step-num {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #E2E8F0;
            color: var(--text-main);
            font-weight: 700;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .stage-step.active .step-num {
            background: var(--primary);
            color: #FFFFFF;
        }

        .step-content h4 {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 2px;
        }

        .step-content p {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* ── Tab 3: Memory Manager ─────────────────────────────── */
        .memory-container {
            width: 100%;
            height: 100%;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            overflow-y: auto;
        }

        .memory-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .memory-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1rem;
        }

        .memory-entry-card {
            background: #FFFFFF;
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }

        .memory-entry-card:hover {
            border-color: var(--primary-border);
            box-shadow: var(--shadow-md);
        }

        .entry-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .entry-word {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 16px;
            font-weight: 800;
            color: var(--primary);
        }

        .entry-context {
            font-size: 13px;
            color: var(--text-body);
            background: #F8FAFC;
            padding: 8px 12px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
        }

        .entry-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
        }

        .delete-btn {
            background: transparent;
            border: none;
            color: #EF4444;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .delete-btn:hover {
            text-decoration: underline;
        }

        /* ── Floating Toast Notification ──────────────────────── */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #0F172A;
            color: #FFFFFF;
            padding: 12px 20px;
            border-radius: var(--radius-md);
            font-size: 13px;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            gap: 8px;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.25s ease;
            z-index: 1000;
            pointer-events: none;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="header-left">
            <div class="brand-logo">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
            </div>
            <div class="brand-info">
                <h1>FlowEdit</h1>
                <div class="tagline">Lifelong Pronunciation Adaptation • Fine-Tuned XTTS ($d=1024$)</div>
            </div>
        </div>

        <div class="header-center">
            <button class="nav-tab active" onclick="switchTab('studio')">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                Synthesis Studio
            </button>
            <button class="nav-tab" onclick="switchTab('learner')">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                Pronunciation Learner
            </button>
            <button class="nav-tab" onclick="switchTab('memory')">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
                Associative Memory (<span id="memCountHeader">4</span>)
            </button>
            <button class="nav-tab" onclick="switchTab('transcribe')">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M19 11a7 7 0 0 1-7 7m0 0a7 7 0 0 1-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 0 0 3-3V5a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3z"/></svg>
                Speech to Text
            </button>
        </div>

        <div class="header-right">
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>XTTS-v2 CUDA Online</span>
            </div>
        </div>
    </header>

    <!-- Tab 1: Synthesis Studio -->
    <div id="tab-studio" class="tab-content active app-container">
        <div class="studio-layout">
            <!-- Sidebar Controls -->
            <aside class="sidebar">
                <!-- Model Selection -->
                <div>
                    <div class="sidebar-section-title">
                        <span>Active Model Backbone</span>
                    </div>
                    <div class="model-picker">
                        <div class="model-option selected" id="opt-fine" onclick="toggleModel('finetuned')">
                            <div class="model-option-left">
                                <div class="custom-checkbox">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                                </div>
                                <span class="model-name">Our Fine-Tuned Model</span>
                            </div>
                            <span class="model-pill">Active</span>
                        </div>
                        <div class="model-option" id="opt-base" onclick="toggleModel('base')">
                            <div class="model-option-left">
                                <div class="custom-checkbox">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                                </div>
                                <span class="model-name">Base Model Baseline</span>
                            </div>
                            <span class="model-pill base">Stock</span>
                        </div>
                    </div>
                </div>

                <!-- Speaker Preset Selection -->
                <div>
                    <div class="sidebar-section-title">
                        <span>Speaker Voice</span>
                    </div>
                    <div class="voice-list" id="voiceList">
                        <div class="voice-card selected" id="speaker-female" onclick="selectSpeaker('female', this)">
                            <div class="voice-card-left">
                                <div class="voice-avatar">F</div>
                                <div class="voice-info">
                                    <span class="voice-title">Blessing</span>
                                    <span class="voice-desc">Female • Neutral Studio</span>
                                </div>
                            </div>
                            <button class="voice-play-btn" onclick="previewSpeaker(event, 'female')" title="Preview Voice">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="m11.596 8.697-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/></svg>
                            </button>
                        </div>
                        <div class="voice-card" id="speaker-male" onclick="selectSpeaker('male', this)">
                            <div class="voice-card-left">
                                <div class="voice-avatar">M</div>
                                <div class="voice-info">
                                    <span class="voice-title">Michael</span>
                                    <span class="voice-desc">Male • Clear Expressive</span>
                                </div>
                            </div>
                            <button class="voice-play-btn" onclick="previewSpeaker(event, 'male')" title="Preview Voice">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="m11.596 8.697-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/></svg>
                            </button>
                        </div>
                        <!-- Dynamic Custom Cloned Voice Card -->
                        <div class="voice-card" id="customVoiceCard" style="display: none; border-color: #10B981;" onclick="selectCustomSpeaker(this)">
                            <div class="voice-card-left">
                                <div class="voice-avatar" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: #FFFFFF;">U</div>
                                <div class="voice-info">
                                    <span class="voice-title" id="customVoiceCardTitle" style="max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">My Cloned Voice</span>
                                    <span class="voice-desc" style="color: #059669; font-weight: 600;">✓ Custom Upload Active</span>
                                </div>
                            </div>
                            <button class="voice-play-btn" onclick="previewCustomVoice(event)" title="Listen to Uploaded Sample">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="m11.596 8.697-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/></svg>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Custom Voice Cloning Upload -->
                <div>
                    <div class="sidebar-section-title">
                        <span>Voice Cloning Upload</span>
                    </div>
                    <div class="upload-voice-box" id="uploadVoiceBox" onclick="document.getElementById('customVoiceFile').click()">
                        <input type="file" id="customVoiceFile" accept="audio/*" style="display:none;" onchange="onCustomVoiceUploaded(event)">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="margin: 0 auto 6px; display:block; color: var(--primary);"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                        <span id="customVoiceLabel">Upload .WAV Reference Voice</span>
                    </div>
                    <div id="customVoiceStatusBanner" style="display: none; margin-top: 8px; padding: 8px 12px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: var(--radius-sm); font-size: 11px; color: #065F46; line-height: 1.4;">
                        <strong>✓ Voice Uploaded:</strong> <span id="customVoiceFileName"></span>
                        <div style="margin-top: 4px; display: flex; gap: 8px;">
                            <a href="javascript:void(0)" onclick="document.getElementById('customVoiceFile').click()" style="color: #059669; font-weight: 700; text-decoration: underline;">Replace</a>
                            <a href="javascript:void(0)" onclick="clearCustomVoice()" style="color: #DC2626; font-weight: 700; text-decoration: underline;">Remove</a>
                        </div>
                    </div>
                </div>
            </aside>

            <!-- Main Studio Workspace -->
            <main class="workspace">
                <div class="prompt-card">
                    <div class="prompt-header">
                        <div class="prompt-title">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                            Carrier Text Prompt
                        </div>
                        <div class="sample-chips">
                            <span style="font-size:11px; font-weight:600; color:var(--text-muted);">Quick Samples:</span>
                            <span class="sample-chip" onclick="setPrompt('The pipes in the old building were completely made of lead.')">Lead (Metal)</span>
                            <span class="sample-chip" onclick="setPrompt('The doctor prescribed hydromorphone 2mg for acute pain.')">Hydromorphone</span>
                            <span class="sample-chip" onclick="setPrompt('Dr. Siobhan administered the prophylactic dose.')">Siobhan</span>
                        </div>
                    </div>

                    <textarea id="promptInput" class="text-input-area" placeholder="Please enter a sentence to synthesize..."></textarea>

                    <div class="prompt-footer">
                        <div class="char-counter" id="charCount">0 characters</div>
                        <div class="action-group">
                            <label style="display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600; color:var(--text-muted); cursor:pointer; margin-right:8px;">
                                <input type="checkbox" id="streamAudioToggle" checked style="accent-color:var(--primary); width:15px; height:15px;">
                                Stream Audio
                            </label>
                            <button class="btn btn-secondary" onclick="document.getElementById('promptInput').value=''; updateCharCount();">Clear</button>
                            <button class="btn btn-primary" id="synthBtn" onclick="runSynthesis()">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                                Generate Speech
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Outputs Section -->
                <div class="outputs-container" id="outputsGrid">
                    <!-- Cards injected dynamically -->
                </div>
            </main>
        </div>
    </div>

    <!-- Tab 2: Pronunciation Learner -->
    <div id="tab-learner" class="tab-content app-container">
        <div class="learner-container">
            <div class="learner-card">
                <div class="prompt-title" style="margin-bottom:0.5rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" stroke="%23DC2626" stroke-width="2.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                    FlowEdit Pronunciation Learning Engine
                </div>
                <p style="font-size:13px; color:var(--text-muted); line-height:1.5;">
                    Correct pronunciations using either ground-truth reference audio (FlowEdit 3-stage optimization) or deterministic phonetic respelling (stored in S3).
                </p>

                <!-- Correction Mode Toggle -->
                <div style="display:flex; gap:8px; margin:0.5rem 0 0.75rem; background:rgba(0,0,0,0.04); padding:4px; border-radius:8px; width:fit-content;">
                    <button type="button" id="modeAudioBtn" style="background:var(--primary); color:#fff; font-weight:700; border:none; padding:7px 14px; font-size:12px; border-radius:6px; cursor:pointer;" onclick="setCorrectionMode('audio')">
                        🎙️ Audio Mode (Reference WAV)
                    </button>
                    <button type="button" id="modeSpellBtn" style="background:transparent; color:var(--text-muted); font-weight:600; border:none; padding:7px 14px; font-size:12px; border-radius:6px; cursor:pointer;" onclick="setCorrectionMode('spell')">
                        ✍️ Spell Mode (Phonetic Respelling / S3)
                    </button>
                </div>

                <div style="display:flex; flex-direction:column; gap:0.75rem;">
                    <label style="font-size:12px; font-weight:700; color:var(--text-main);">Full Carrier Sentence</label>
                    <input type="text" id="correctSentence" class="text-input-area" style="min-height:46px; height:46px;" value="She read the book yesterday.">
                </div>

                <div style="display:flex; flex-direction:column; gap:0.75rem;">
                    <label style="font-size:12px; font-weight:700; color:var(--text-main);">Target Mispronounced Word</label>
                    <input type="text" id="correctWord" class="text-input-area" style="min-height:46px; height:46px;" value="read">
                </div>

                <!-- Spell Mode Input (Shown when in Spell Mode) -->
                <div id="spellAsContainer" style="display:none; flex-direction:column; gap:0.75rem;">
                    <label style="font-size:12px; font-weight:700; color:var(--text-main);">Spell it as (Phonetic spelling, e.g. 'red' for 'read')</label>
                    <input type="text" id="correctSpellAs" class="text-input-area" style="min-height:46px; height:46px;" placeholder="e.g. red" value="red">
                    <div style="font-size:11px; color:var(--text-muted); line-height:1.4;">
                        Uses shared homograph context resolver to classify sense (e.g. past vs present) and stores directly in S3.
                    </div>
                </div>

                <!-- Audio Mode Upload (Shown when in Audio Mode) -->
                <div id="refAudioContainer" style="display:flex; flex-direction:column; gap:0.75rem;">
                    <label style="font-size:12px; font-weight:700; color:var(--text-main);">Reference Correct Pronunciation Audio (.WAV)</label>
                    <div class="upload-voice-box" onclick="document.getElementById('refAudioFile').click()">
                        <input type="file" id="refAudioFile" accept="audio/*" style="display:none;" onchange="onRefAudioUploaded(event)">
                        <span id="refAudioLabel">Select / Upload Ground-Truth WAV</span>
                    </div>
                </div>

                <button class="btn btn-primary" id="learnBtn" style="margin-top:0.5rem;" onclick="runPronunciationCorrection()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
                    <span id="learnBtnText">Optimize & Learn Pronunciation</span>
                </button>
            </div>

            <!-- Stages Feedback -->
            <div class="learner-card">
                <div class="prompt-title">Pipeline Diagnostics & Progress</div>
                <div class="stage-indicator">
                    <div class="stage-step" id="stage1">
                        <div class="step-num">1</div>
                        <div class="step-content">
                            <h4>Stage 1: Whisper Forced Alignment</h4>
                            <p id="stage1-text">Awaiting input audio and target text...</p>
                        </div>
                    </div>
                    <div class="stage-step" id="stage2">
                        <div class="step-num">2</div>
                        <div class="step-content">
                            <h4>Stage 2: 50-Step Latent Input Optimization ($\delta^*$)</h4>
                            <p id="stage2-text">Differentiable teacher-forced CE loss over Discrete VAE codes.</p>
                        </div>
                    </div>
                    <div class="stage-step" id="stage3">
                        <div class="step-num">3</div>
                        <div class="step-content">
                            <h4>Stage 3: Modern Hopfield Associative Memory Storage</h4>
                            <p id="stage3-text">Key-Value pair ($K_i, V_i \in \mathbb{R}^{1024}$) write & threshold calibration.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Tab 3: Memory Manager -->
    <div id="tab-memory" class="tab-content app-container">
        <div class="memory-container">
            <!-- Section 1: S3 Cloud Phonetic Spelling Dictionary -->
            <div style="background:#FFFFFF; border:1px solid var(--border-subtle); border-radius:var(--radius-md); padding:1.5rem; display:flex; flex-direction:column; gap:1.25rem; box-shadow:var(--shadow-sm);">
                <div class="memory-header" style="border-bottom:1px solid var(--border-subtle); padding-bottom:1rem;">
                    <div>
                        <div style="display:flex; align-items:center; gap:0.75rem;">
                            <h2 style="font-family:'Plus Jakarta Sans',sans-serif; font-size:18px; font-weight:800; color:var(--text-main);">
                                ☁️ Amazon S3 Phonetic Spelling Dictionary
                            </h2>
                            <span id="s3BucketBadge" class="status-badge" style="background:#EFF6FF; border-color:#BFDBFE; color:#1D4ED8; font-size:11px; padding:3px 10px;">
                                Bucket: flowedit-bucket
                            </span>
                        </div>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                            Deterministic pronunciation corrections stored directly in S3 (<code id="s3UriCode" style="background:#F1F5F9; padding:2px 6px; border-radius:4px; font-size:11px;">s3://flowedit-bucket/corrections/spelling_dictionary.json</code>).
                        </p>
                    </div>
                    <div style="display:flex; gap:0.5rem;">
                        <button class="btn btn-secondary" style="font-size:12px; padding:6px 12px;" onclick="loadS3SpellingEntries(true)">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                            Refresh S3
                        </button>
                        <button class="btn btn-secondary" style="color:#EF4444; border-color:#FCA5A5; font-size:12px; padding:6px 12px;" onclick="clearAllS3Spelling()">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                            Clear S3 Dictionary
                        </button>
                    </div>
                </div>

                <div id="s3SpellingGrid" class="memory-grid">
                    <!-- S3 Word Cards Injected Dynamically -->
                </div>
            </div>

            <!-- Section 2: Modern Hopfield Pronunciation Memory -->
            <div style="background:#FFFFFF; border:1px solid var(--border-subtle); border-radius:var(--radius-md); padding:1.5rem; display:flex; flex-direction:column; gap:1.25rem; box-shadow:var(--shadow-sm);">
                <div class="memory-header" style="border-bottom:1px solid var(--border-subtle); padding-bottom:1rem;">
                    <div>
                        <h2 style="font-family:'Plus Jakarta Sans',sans-serif; font-size:18px; font-weight:800; color:var(--text-main);">
                            🧠 Modern Hopfield Pronunciation Memory
                        </h2>
                        <p style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                            Stored continuous vector corrections ($d=1024$) learned via 3-stage optimization and applied automatically at inference time.
                        </p>
                    </div>
                    <button class="btn btn-secondary" style="color:#EF4444; border-color:#FCA5A5; font-size:12px; padding:6px 12px;" onclick="clearAllMemory()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        Clear Hopfield Memory
                    </button>
                </div>

                <div class="memory-grid" id="memoryGrid">
                    <!-- Memory Cards Injected Dynamically -->
                </div>
            </div>
        </div>
    </div>

    <!-- Tab 4: Speech to Text (Transcribe) -->
    <div id="tab-transcribe" class="tab-content app-container">
        <div style="width:100%; max-width:1100px; margin:0 auto; padding:2rem; display:flex; flex-direction:column; gap:1.5rem; overflow-y:auto;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h2 style="font-family:'Plus Jakarta Sans',sans-serif; font-size:22px; font-weight:800; color:var(--text-main); display:flex; align-items:center; gap:0.5rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="var(--primary)" stroke-width="2.5" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                        Whisper Speech-to-Text Transcriber
                    </h2>
                    <p style="font-size:13px; color:var(--text-muted); margin-top:4px;">Upload speech audio in any format (WAV, MP3, M4A, OGG, FLAC) to transcribe into text with high accuracy.</p>
                </div>
                <div class="status-badge" style="background:#EEF2FF; border-color:#C7D2FE; color:#4F46E5;">
                    <span class="status-dot" style="background:#4F46E5; box-shadow:0 0 8px #4F46E5;"></span>
                    <span>Whisper Engine Ready</span>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1.2fr; gap:1.5rem;">
                <!-- Left: Upload & Settings Card -->
                <div class="prompt-card" style="gap:1.25rem;">
                    <div class="prompt-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                        Audio Input
                    </div>

                    <!-- Drop Zone -->
                    <div id="transcribeDropZone" class="upload-voice-box" style="padding:2rem 1.5rem; display:flex; flex-direction:column; align-items:center; gap:0.75rem; border-width:2px;" onclick="document.getElementById('transcribeAudioInput').click()">
                        <input type="file" id="transcribeAudioInput" accept="audio/*" style="display:none;" onchange="onTranscribeAudioSelected(event)">
                        <div style="width:48px; height:48px; border-radius:50%; background:var(--primary-bg); display:flex; align-items:center; justify-content:center; color:var(--primary);">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
                        </div>
                        <div style="text-align:center;">
                            <span id="transcribeFileName" style="font-size:14px; font-weight:700; color:var(--text-main); display:block;">Click to browse or drop audio here</span>
                            <span style="font-size:11px; color:var(--text-muted);">Supports WAV, MP3, M4A, OGG, FLAC, WebM</span>
                        </div>
                    </div>

                    <!-- Audio Preview Player -->
                    <audio id="transcribeAudioPreview" controls style="display:none; width:100%; border-radius:var(--radius-sm);"></audio>

                    <!-- Language Selection -->
                    <div style="display:flex; flex-direction:column; gap:0.5rem;">
                        <label style="font-size:12px; font-weight:700; color:var(--text-main);">Audio Spoken Language</label>
                        <select id="transcribeLanguage" class="text-input-area" style="min-height:42px; height:42px; padding:0 0.75rem; cursor:pointer;">
                            <option value="">Auto-Detect Language</option>
                            <option value="en">English (en)</option>
                            <option value="hi">Hindi (hi)</option>
                            <option value="mr">Marathi (mr)</option>
                            <option value="es">Spanish (es)</option>
                            <option value="fr">French (fr)</option>
                            <option value="de">German (de)</option>
                            <option value="it">Italian (it)</option>
                            <option value="pt">Portuguese (pt)</option>
                            <option value="ja">Japanese (ja)</option>
                            <option value="zh">Chinese (zh)</option>
                        </select>
                    </div>

                    <!-- Hopfield Associative Memory Biasing & Correction Toggle -->
                    <div style="display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1rem; background:var(--primary-bg); border:1.5px solid var(--primary-border); border-radius:var(--radius-md);">
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <input type="checkbox" id="transcribeUseMemory" checked style="accent-color:var(--primary); width:18px; height:18px; cursor:pointer;">
                            <div>
                                <label for="transcribeUseMemory" style="font-size:13px; font-weight:700; color:var(--text-main); cursor:pointer; display:block;">Hopfield Memory Biasing</label>
                                <span style="font-size:11px; color:var(--text-muted);">Auto-correct phonetic misspellings to learned words</span>
                            </div>
                        </div>
                        <span id="transcribeMemBadge" style="font-size:11px; font-weight:700; color:var(--primary); background:#FFFFFF; padding:3px 8px; border-radius:var(--radius-full); border:1px solid var(--primary-border);">Active (<span id="transcribeMemCount">0</span> words)</span>
                    </div>

                    <div style="display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1rem; background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:var(--radius-md);">
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <input type="checkbox" id="transcribeStreamToggle" checked style="accent-color:var(--primary); width:18px; height:18px; cursor:pointer;">
                            <div>
                                <label for="transcribeStreamToggle" style="font-size:13px; font-weight:700; color:var(--text-main); cursor:pointer; display:block;">Live Streaming (SSE)</label>
                                <span style="font-size:11px; color:var(--text-muted);">Stream transcription segments and corrections in real time</span>
                            </div>
                        </div>
                        <span style="font-size:10px; font-weight:800; color:#059669; background:#ECFDF5; border:1px solid #A7F3D0; padding:2px 8px; border-radius:var(--radius-full);">REALTIME</span>
                    </div>

                    <button class="btn btn-primary" id="btnTranscribe" onclick="runTranscription()" style="width:100%; margin-top:0.5rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                        Transcribe Audio to Text
                    </button>
                </div>

                <!-- Right: Output Card -->
                <div class="prompt-card" style="display:flex; flex-direction:column; gap:1.25rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="prompt-title">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                            Transcribed Output
                        </div>
                        <div id="transcribeMeta" style="display:none; align-items:center; gap:0.5rem;">
                            <span id="transcribeLangBadge" style="font-size:11px; font-weight:700; background:var(--primary-bg); color:var(--primary); padding:3px 8px; border-radius:var(--radius-full); text-transform:uppercase;">EN</span>
                            <span id="transcribeDurationBadge" style="font-size:11px; font-weight:600; color:var(--text-muted);">0.0s</span>
                        </div>
                    </div>

                    <!-- Output Text Area -->
                    <div id="transcribeEmptyState" style="min-height:160px; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1.5px dashed var(--border-subtle); border-radius:var(--radius-md); color:var(--text-muted); text-align:center; padding:2rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="none" stroke="var(--border-strong)" stroke-width="1.5" viewBox="0 0 24 24" style="margin-bottom:0.75rem;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                        <p style="font-size:13px; font-weight:500;">No transcription yet</p>
                        <p style="font-size:11px; color:var(--text-light); margin-top:2px;">Upload an audio file on the left and click "Transcribe"</p>
                    </div>

                    <div id="transcribeResultContainer" style="display:none; flex-direction:column; gap:1rem;">
                        <!-- Hopfield Memory Correction Alert Banner -->
                        <div id="hopfieldCorrectionBanner" style="display:none; flex-direction:column; gap:0.4rem; padding:0.75rem 1rem; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:var(--radius-md);">
                            <div style="display:flex; align-items:center; gap:0.5rem; color:#15803D; font-size:12px; font-weight:700;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                                Hopfield Memory Restored Canonical Spelling
                            </div>
                            <div id="hopfieldCorrectionList" style="font-size:12px; color:#166534; display:flex; flex-wrap:wrap; gap:0.4rem;"></div>
                        </div>

                        <div style="position:relative;">
                            <textarea id="transcribeResultText" class="text-input-area" style="min-height:150px; font-size:15px; line-height:1.6;" readonly></textarea>
                            <button class="btn btn-secondary" onclick="copyTranscription()" style="position:absolute; top:8px; right:8px; padding:4px 10px; font-size:11px;">
                                Copy Text
                            </button>
                        </div>

                        <div style="display:flex; gap:0.75rem;">
                            <button class="btn btn-secondary" style="flex:1; font-size:12px;" onclick="sendToStudio()">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
                                Send to Synthesis Studio
                            </button>
                            <button class="btn btn-secondary" style="flex:1; font-size:12px;" onclick="sendToLearner()">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="2"/></svg>
                                Send to Pronunciation Learner
                            </button>
                        </div>

                        <!-- Segment Timeline -->
                        <div id="transcribeSegmentsWrapper" style="display:none; flex-direction:column; gap:0.5rem; margin-top:0.5rem;">
                            <span style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted);">Timed Segments</span>
                            <div id="transcribeSegmentsList" style="max-height:160px; overflow-y:auto; display:flex; flex-direction:column; gap:0.4rem; padding-right:4px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div class="toast" id="toast"></div>

    <script>
        let _selectedModels = ['finetuned'];
        let _selectedSpeaker = 'female';
        let _customSpeakerFile = null;
        let _refAudioFile = null;

        document.getElementById('promptInput').addEventListener('input', updateCharCount);

        function updateCharCount() {
            const val = document.getElementById('promptInput').value;
            document.getElementById('charCount').textContent = val.length + ' characters';
        }

        function switchTab(tabId) {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (tabId === 'studio') {
                document.querySelectorAll('.nav-tab')[0].classList.add('active');
                document.getElementById('tab-studio').classList.add('active');
            } else if (tabId === 'learner') {
                document.querySelectorAll('.nav-tab')[1].classList.add('active');
                document.getElementById('tab-learner').classList.add('active');
            } else if (tabId === 'memory') {
                document.querySelectorAll('.nav-tab')[2].classList.add('active');
                document.getElementById('tab-memory').classList.add('active');
                loadMemoryEntries();
                loadS3SpellingEntries();
            } else if (tabId === 'transcribe') {
                document.querySelectorAll('.nav-tab')[3].classList.add('active');
                document.getElementById('tab-transcribe').classList.add('active');
            }
        }

        function setPrompt(text) {
            document.getElementById('promptInput').value = text;
            updateCharCount();
        }

        function toggleModel(modelKey) {
            if (_selectedModels.includes(modelKey)) {
                if (_selectedModels.length > 1) {
                    _selectedModels = _selectedModels.filter(m => m !== modelKey);
                }
            } else {
                _selectedModels.push(modelKey);
            }
            
            document.getElementById('opt-fine').classList.toggle('selected', _selectedModels.includes('finetuned'));
            document.getElementById('opt-base').classList.toggle('selected', _selectedModels.includes('base'));
        }

        let _uploadedCustomVoice = null;
        let _customVoiceAudioUrl = null;

        function selectSpeaker(spk, element) {
            _selectedSpeaker = spk;
            _customSpeakerFile = null;
            document.querySelectorAll('.voice-card').forEach(c => c.classList.remove('selected'));
            if (element) element.classList.add('selected');
            showToast(`Selected ${spk === 'female' ? 'Blessing (Female)' : 'Michael (Male)'} voice.`);
        }

        function selectCustomSpeaker(element) {
            if (!_uploadedCustomVoice) return;
            _selectedSpeaker = 'custom';
            _customSpeakerFile = _uploadedCustomVoice.file;
            document.querySelectorAll('.voice-card').forEach(c => c.classList.remove('selected'));
            if (element) element.classList.add('selected');
            showToast(`Selected custom voice: ${_uploadedCustomVoice.name}`);
        }

        function previewSpeaker(event, name) {
            event.stopPropagation();
            const audio = new Audio(`/api/speakers/${name}/audio`);
            audio.play().catch(e => showToast('Could not preview voice: ' + e));
        }

        function previewCustomVoice(event) {
            event.stopPropagation();
            if (_customVoiceAudioUrl) {
                const audio = new Audio(_customVoiceAudioUrl);
                audio.play().catch(e => showToast('Could not preview custom voice: ' + e));
            }
        }

        function onCustomVoiceUploaded(event) {
            const file = event.target.files[0];
            if (file) {
                _uploadedCustomVoice = { file: file, name: file.name };
                _customSpeakerFile = file;
                _selectedSpeaker = 'custom';
                if (_customVoiceAudioUrl) URL.revokeObjectURL(_customVoiceAudioUrl);
                _customVoiceAudioUrl = URL.createObjectURL(file);

                // Update custom voice card in list
                const card = document.getElementById('customVoiceCard');
                card.style.display = 'flex';
                document.getElementById('customVoiceCardTitle').textContent = file.name;
                
                // Select custom voice card
                document.querySelectorAll('.voice-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');

                // Update upload box and status banner
                document.getElementById('customVoiceLabel').textContent = 'Change / Replace Voice';
                document.getElementById('customVoiceFileName').textContent = file.name;
                document.getElementById('customVoiceStatusBanner').style.display = 'block';

                showToast(`✓ Voice "${file.name}" uploaded & selected as active voice!`);
            }
        }

        function clearCustomVoice() {
            _uploadedCustomVoice = null;
            _customSpeakerFile = null;
            if (_customVoiceAudioUrl) {
                URL.revokeObjectURL(_customVoiceAudioUrl);
                _customVoiceAudioUrl = null;
            }
            document.getElementById('customVoiceFile').value = '';
            document.getElementById('customVoiceCard').style.display = 'none';
            document.getElementById('customVoiceStatusBanner').style.display = 'none';
            document.getElementById('customVoiceLabel').textContent = 'Upload .WAV Reference Voice';
            
            // Revert selection to Blessing
            const blessingCard = document.getElementById('speaker-female');
            if (blessingCard) selectSpeaker('female', blessingCard);
            showToast('Custom voice removed. Reverted to Blessing.');
        }

        function onRefAudioUploaded(event) {
            const file = event.target.files[0];
            if (file) {
                _refAudioFile = file;
                document.getElementById('refAudioLabel').textContent = file.name;
                showToast('Reference audio loaded: ' + file.name);
            }
        }

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), 3500);
        }

        /* ── Synthesis Execution ───────────────────────────────── */
        async function runSynthesis() {
            const text = document.getElementById('promptInput').value.trim();
            if (!text) {
                showToast('Please enter text to synthesize.');
                return;
            }

            const btn = document.getElementById('synthBtn');
            btn.disabled = true;
            btn.innerHTML = 'Synthesizing...';

            const grid = document.getElementById('outputsGrid');
            grid.innerHTML = '';

            for (const modelKey of _selectedModels) {
                const isFine = modelKey === 'finetuned';
                const endpoint = isFine ? '/api/synthesize' : '/api/baseline';
                const cardTitle = isFine ? 'Fine-Tuned XTTS ($d=1024$)' : 'Base Model Baseline';
                const tagClass = isFine ? 'finetuned' : '';

                const card = document.createElement('div');
                card.className = `output-card ${tagClass}`;
                card.id = `card-${modelKey}`;
                card.innerHTML = `
                    <div class="output-card-header">
                        <div class="output-tag ${tagClass}">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
                            ${cardTitle}
                        </div>
                        <span style="font-size:11px; font-weight:700; color:var(--text-muted);" id="status-${modelKey}">Synthesizing...</span>
                    </div>
                    <div class="metrics-row" id="metrics-${modelKey}" style="display:none;">
                        <div class="metric-item"><span class="metric-label">Latency</span><span class="metric-value" id="ttfa-${modelKey}">—</span></div>
                        <div class="metric-item"><span class="metric-label">Duration</span><span class="metric-value" id="dur-${modelKey}">—</span></div>
                        <div class="metric-item"><span class="metric-label">Memory</span><span class="metric-value" id="mem-${modelKey}">Active</span></div>
                    </div>
                    <div class="player-row">
                        <audio id="audio-${modelKey}" controls style="display:none;"></audio>
                        <a id="dl-${modelKey}" class="dl-button" style="display:none;" title="Download WAV">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        </a>
                    </div>
                `;
                grid.appendChild(card);

                try {
                    const formData = new FormData();
                    formData.append('text', text);
                    formData.append('language', 'en');
                    formData.append('speaker_name', _selectedSpeaker);
                    if (_customSpeakerFile) {
                        formData.append('speaker_wav', _customSpeakerFile);
                    }

                    const isStreamingAudio = document.getElementById('streamAudioToggle') && document.getElementById('streamAudioToggle').checked;
                    if (isStreamingAudio && isFine) {
                        formData.append('stream', 'true');
                    }

                    const t0 = performance.now();
                    const resp = await fetch(endpoint, { method: 'POST', body: formData });
                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        throw new Error(err.detail || 'Synthesis failed');
                    }

                    let blob;
                    let firstChunkTime = null;
                    if (isStreamingAudio && isFine && resp.body) {
                        const reader = resp.body.getReader();
                        const chunks = [];
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            if (firstChunkTime === null) {
                                firstChunkTime = ((performance.now() - t0) / 1000).toFixed(2);
                                const st = document.getElementById(`status-${modelKey}`);
                                if (st) {
                                    st.textContent = '⚡ Streaming...';
                                    st.style.color = '#3B82F6';
                                }
                            }
                            chunks.push(value);
                        }
                        blob = new Blob(chunks, { type: 'audio/wav' });
                    } else {
                        blob = await resp.blob();
                    }

                    const elapsedSec = firstChunkTime || ((performance.now() - t0) / 1000).toFixed(2);
                    const audioUrl = URL.createObjectURL(blob);

                    const audioEl = document.getElementById(`audio-${modelKey}`);
                    const dlEl = document.getElementById(`dl-${modelKey}`);
                    const statusEl = document.getElementById(`status-${modelKey}`);
                    const metricsEl = document.getElementById(`metrics-${modelKey}`);
                    const isMemActive = resp.headers.get('X-Memory-Active') === 'True';

                    audioEl.src = audioUrl;
                    audioEl.style.display = 'block';
                    dlEl.href = audioUrl;
                    dlEl.download = `${modelKey}_synthesized.wav`;
                    dlEl.style.display = 'flex';

                    document.getElementById(`ttfa-${modelKey}`).textContent = (firstChunkTime ? `⚡ ${firstChunkTime}s (TTFA)` : `${elapsedSec}s`);
                    document.getElementById(`dur-${modelKey}`).textContent = (blob.size / (24000 * 2)).toFixed(2) + 's';
                    document.getElementById(`mem-${modelKey}`).textContent = isMemActive ? '✓ Refined' : 'Standard';
                    metricsEl.style.display = 'flex';
                    statusEl.textContent = isStreamingAudio ? '✓ Stream Complete' : '✓ Ready';
                    statusEl.style.color = '#22C55E';

                    audioEl.play().catch(()=>{});

                } catch (e) {
                    document.getElementById(`status-${modelKey}`).textContent = 'Failed';
                    document.getElementById(`status-${modelKey}`).style.color = '#DC2626';
                    showToast(`Synthesis Error (${modelKey}): ` + e.message);
                }
            }

            btn.disabled = false;
            btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg> Generate Speech`;
        }

        /* ── Pronunciation Correction Execution ────────────────── */
        let _currentCorrectionMode = 'audio';

        function setCorrectionMode(mode) {
            _currentCorrectionMode = mode;
            const audioBtn = document.getElementById('modeAudioBtn');
            const spellBtn = document.getElementById('modeSpellBtn');
            const refAudioContainer = document.getElementById('refAudioContainer');
            const spellAsContainer = document.getElementById('spellAsContainer');
            const learnBtnText = document.getElementById('learnBtnText');

            if (mode === 'spell') {
                audioBtn.style.background = 'transparent';
                audioBtn.style.color = 'var(--text-muted)';
                audioBtn.style.fontWeight = '600';

                spellBtn.style.background = 'var(--primary)';
                spellBtn.style.color = '#fff';
                spellBtn.style.fontWeight = '700';

                refAudioContainer.style.display = 'none';
                spellAsContainer.style.display = 'flex';
                if (learnBtnText) learnBtnText.textContent = 'Save Phonetic Spelling to S3';
            } else {
                spellBtn.style.background = 'transparent';
                spellBtn.style.color = 'var(--text-muted)';
                spellBtn.style.fontWeight = '600';

                audioBtn.style.background = 'var(--primary)';
                audioBtn.style.color = '#fff';
                audioBtn.style.fontWeight = '700';

                refAudioContainer.style.display = 'flex';
                spellAsContainer.style.display = 'none';
                if (learnBtnText) learnBtnText.textContent = 'Optimize & Learn Pronunciation';
            }
        }

        async function runPronunciationCorrection() {
            const sentence = document.getElementById('correctSentence').value.trim();
            const word = document.getElementById('correctWord').value.trim();

            if (!sentence || !word) {
                showToast('Please enter both carrier sentence and target word.');
                return;
            }

            if (_currentCorrectionMode === 'spell') {
                const spellAs = (document.getElementById('correctSpellAs').value || '').trim();
                if (!spellAs) {
                    showToast('Please enter the phonetic spelling (e.g. red for read).');
                    return;
                }

                const btn = document.getElementById('learnBtn');
                btn.disabled = true;
                btn.textContent = 'Saving Phonetic Spelling to S3...';

                const st1 = document.getElementById('stage1');
                const st2 = document.getElementById('stage2');
                const st3 = document.getElementById('stage3');

                st1.classList.add('active');
                document.getElementById('stage1-text').textContent = 'Classifying word context & syntactic sense...';

                try {
                    const formData = new FormData();
                    formData.append('text', sentence);
                    formData.append('target_word', word);
                    formData.append('mode', 'spell');
                    formData.append('spell_as', spellAs);

                    st2.classList.add('active');
                    document.getElementById('stage2-text').textContent = `Mapping '${word}' ➔ '${spellAs}'...`;

                    const resp = await fetch('/api/correct', { method: 'POST', body: formData });
                    const res = await resp.json();
                    if (!resp.ok) {
                        throw new Error(res.detail || 'Spelling correction failed');
                    }

                    st3.classList.add('active');
                    document.getElementById('stage1-text').textContent = `✓ Sense: ${res.sense_display || res.sense_id}`;
                    document.getElementById('stage2-text').textContent = `✓ Phonetic mapping: '${res.word}' ➔ '${res.spell_as}'`;
                    document.getElementById('stage3-text').textContent = `✓ S3 Persistence: ${res.s3_status === 'synced' ? 'Synchronized to S3' : 'Held in memory (ready for S3 link)'}`;

                    showToast(`✓ Phonetic spelling for '${res.word}' saved to S3!`);
                    loadS3SpellingEntries();
                } catch (e) {
                    showToast('Spelling Correction Failed: ' + e.message);
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> <span id="learnBtnText">Save Phonetic Spelling to S3</span>`;
                }
                return;
            }

            if (!_refAudioFile) {
                showToast('Please upload a ground-truth reference audio WAV.');
                return;
            }

            const btn = document.getElementById('learnBtn');
            btn.disabled = true;
            btn.textContent = 'Running Stage 1 & 2 Optimization...';

            const st1 = document.getElementById('stage1');
            const st2 = document.getElementById('stage2');
            const st3 = document.getElementById('stage3');

            st1.classList.add('active');
            document.getElementById('stage1-text').textContent = 'Aligning word boundaries with Whisper base model...';

            try {
                const formData = new FormData();
                formData.append('text', sentence);
                formData.append('target_word', word);
                formData.append('ref_audio', _refAudioFile);
                formData.append('speaker_name', _selectedSpeaker);
                if (_customSpeakerFile) {
                    formData.append('speaker_wav', _customSpeakerFile);
                }

                st2.classList.add('active');
                document.getElementById('stage2-text').textContent = 'Computing 50-step Adam latent optimization on δ*...';

                const resp = await fetch('/api/correct', { method: 'POST', body: formData });
                const res = await resp.json();

                if (!resp.ok) {
                    throw new Error(res.detail || 'Correction failed');
                }

                st3.classList.add('active');
                document.getElementById('stage1-text').textContent = `✓ Grounded '${res.word}' successfully.`;
                document.getElementById('stage2-text').textContent = `✓ Optimized δ* (Final Loss: ${res.final_loss ? res.final_loss.toFixed(4) : 'converged'}).`;
                document.getElementById('stage3-text').textContent = `✓ Stored in Modern Hopfield Memory (Total Entries: ${res.memory_size}).`;

                document.getElementById('memCountHeader').textContent = res.memory_size;
                showToast(`✓ Correction for '${res.word}' stored successfully!`);

            } catch (e) {
                showToast('Correction Failed: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Optimize & Learn Pronunciation`;
            }
        }

        /* ── Memory Manager Functions ──────────────────────────── */
        async function loadMemoryEntries() {
            try {
                const resp = await fetch('/api/memory');
                const data = await resp.json();
                document.getElementById('memCountHeader').textContent = data.size;

                const grid = document.getElementById('memoryGrid');
                grid.innerHTML = '';

                if (data.corrections.length === 0) {
                    grid.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding:3rem; color:var(--text-muted);">No pronunciation corrections stored in memory.</div>`;
                    return;
                }

                data.corrections.forEach(entry => {
                    const card = document.createElement('div');
                    card.className = 'memory-entry-card';
                    card.innerHTML = `
                        <div class="entry-top">
                            <span class="entry-word">${entry.word}</span>
                            <button class="delete-btn" onclick="deleteMemoryWord('${entry.word}')">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                                Delete
                            </button>
                        </div>
                        <div class="entry-context">${entry.carrier || 'No context recorded'}</div>
                        <div class="entry-footer">
                            <span>Access count: <strong>${entry.access_count}</strong></span>
                            <span>Lang: <strong>${entry.language}</strong></span>
                        </div>
                    `;
                    grid.appendChild(card);
                });

            } catch (e) {
                showToast('Failed to load memory: ' + e.message);
            }
        }

        async function deleteMemoryWord(word) {
            if (!confirm(`Delete correction for '${word}' from Hopfield Memory?`)) return;
            try {
                const resp = await fetch(`/api/memory/${encodeURIComponent(word)}`, { method: 'DELETE' });
                if (resp.ok) {
                    showToast(`Deleted '${word}'`);
                    loadMemoryEntries();
                }
            } catch (e) {
                showToast('Deletion error: ' + e.message);
            }
        }

        async function clearAllMemory() {
            if (!confirm('Clear all stored pronunciation corrections? This cannot be undone.')) return;
            try {
                const resp = await fetch('/api/memory/clear', { method: 'POST' });
                if (resp.ok) {
                    showToast('Memory cleared.');
                    loadMemoryEntries();
                }
            } catch (e) {
                showToast('Error clearing memory: ' + e.message);
            }
        }

        /* ── S3 Spelling Store Functions ──────────────────────── */
        async function loadS3SpellingEntries(refresh = false) {
            try {
                const url = refresh ? '/api/s3/entries?refresh=true' : '/api/s3/entries';
                const resp = await fetch(url);
                const data = await resp.json();
                
                const badge = document.getElementById('s3BucketBadge');
                if (badge && data.bucket) {
                    badge.textContent = `Bucket: ${data.bucket}`;
                }
                const uriCode = document.getElementById('s3UriCode');
                if (uriCode && data.s3_uri) {
                    uriCode.textContent = data.s3_uri;
                }

                const grid = document.getElementById('s3SpellingGrid');
                if (!grid) return;
                grid.innerHTML = '';

                if (!data.entries || data.entries.length === 0) {
                    grid.innerHTML = `
                        <div style="grid-column: 1/-1; text-align:center; padding:2rem; color:var(--text-muted); background:var(--bg-page); border-radius:var(--radius-sm);">
                            No phonetic spelling corrections in S3 yet. Use <strong>Spell Mode</strong> in the Pronunciation Learner tab to add one!
                        </div>
                    `;
                    return;
                }

                data.entries.forEach(entry => {
                    const card = document.createElement('div');
                    card.className = 'memory-entry-card';
                    
                    const sensesHtml = (entry.senses || []).map(s => `
                        <div style="margin-top:6px; padding:6px 8px; background:var(--bg-page); border-radius:6px; font-size:11px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                                <span style="font-weight:700; color:var(--text-main);">${s.display_name || s.sense_id}</span>
                                <span style="font-family:'JetBrains Mono',monospace; color:var(--primary); font-weight:700;">➔ ${s.spell_as}</span>
                            </div>
                            <div style="color:var(--text-muted); font-style:italic;">"${s.carrier_text || 'No context'}"</div>
                            ${s.context_clues && s.context_clues.length ? `<div style="color:var(--text-light); margin-top:2px;">Clues: ${s.context_clues.join(', ')}</div>` : ''}
                        </div>
                    `).join('');

                    card.innerHTML = `
                        <div class="entry-top">
                            <div>
                                <span class="entry-word" style="color:#0F172A; font-size:17px;">${entry.word}</span>
                                <span style="margin-left:8px; font-family:'JetBrains Mono',monospace; color:var(--primary); font-weight:700; background:var(--primary-bg); padding:2px 8px; border-radius:4px; font-size:12px;">
                                    ➔ ${entry.default_spell}
                                </span>
                            </div>
                            <button class="delete-btn" title="Delete from S3" onclick="deleteS3SpellingWord('${entry.word}')">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                                Delete
                            </button>
                        </div>
                        <div style="font-size:12px; color:var(--text-muted); font-weight:600; margin-top:4px;">
                            Registered Senses (${(entry.senses || []).length}):
                        </div>
                        <div style="max-height:140px; overflow-y:auto;">
                            ${sensesHtml}
                        </div>
                        <div class="entry-footer" style="margin-top:auto; padding-top:8px; border-top:1px solid var(--border-subtle);">
                            <span>Target: <strong>${entry.default_spell}</strong></span>
                            <span class="status-badge" style="background:#ECFDF5; border-color:#A7F3D0; color:#059669; font-size:10px; padding:2px 6px;">
                                S3 Synced
                            </span>
                        </div>
                    `;
                    grid.appendChild(card);
                });

            } catch (e) {
                console.error('Failed to load S3 spelling entries:', e);
            }
        }

        async function deleteS3SpellingWord(word) {
            if (!confirm(`Delete phonetic spelling for '${word}' from Amazon S3 bucket?`)) return;
            try {
                const resp = await fetch(`/api/s3/entries/${encodeURIComponent(word)}`, { method: 'DELETE' });
                const res = await resp.json();
                if (resp.ok) {
                    showToast(`✓ Deleted '${word}' from S3 bucket!`);
                    loadS3SpellingEntries();
                } else {
                    showToast('Failed to delete: ' + (res.detail || 'Error'));
                }
            } catch (e) {
                showToast('Deletion error: ' + e.message);
            }
        }

        async function clearAllS3Spelling() {
            if (!confirm('Clear ALL phonetic spelling corrections from Amazon S3? This will empty the dictionary in the S3 bucket.')) return;
            try {
                const resp = await fetch('/api/s3/entries', { method: 'DELETE' });
                const res = await resp.json();
                if (resp.ok) {
                    showToast('✓ All entries cleared from S3 bucket.');
                    loadS3SpellingEntries();
                } else {
                    showToast('Failed to clear S3: ' + (res.detail || 'Error'));
                }
            } catch (e) {
                showToast('Error clearing S3 dictionary: ' + e.message);
            }
        }

        // Speech-to-Text Transcription Logic
        let _transcribeFile = null;
        let _transcribeAudioUrl = null;

        function onTranscribeAudioSelected(event) {
            const file = event.target.files[0];
            if (file) {
                _transcribeFile = file;
                document.getElementById('transcribeFileName').textContent = file.name;
                if (_transcribeAudioUrl) URL.revokeObjectURL(_transcribeAudioUrl);
                _transcribeAudioUrl = URL.createObjectURL(file);
                const preview = document.getElementById('transcribeAudioPreview');
                preview.src = _transcribeAudioUrl;
                preview.style.display = 'block';
                showToast(`Loaded audio: ${file.name}`);
            }
        }

        async function runTranscription() {
            if (!_transcribeFile) {
                showToast('Please select or upload an audio file first.');
                return;
            }

            const btn = document.getElementById('btnTranscribe');
            const originalBtnText = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = `
                <svg class="spinner" style="animation:spin 1s linear infinite; width:16px; height:16px; display:inline-block; vertical-align:middle; margin-right:6px;" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" style="opacity:0.25;"></circle><path fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" style="opacity:0.75;"></path></svg>
                Transcribing with Whisper...
            `;

            try {
                const formData = new FormData();
                formData.append('audio', _transcribeFile);
                const lang = document.getElementById('transcribeLanguage').value;
                if (lang) formData.append('language', lang);
                formData.append('include_timestamps', 'true');
                formData.append('use_memory', document.getElementById('transcribeUseMemory').checked);

                const isStreaming = document.getElementById('transcribeStreamToggle') && document.getElementById('transcribeStreamToggle').checked;

                if (isStreaming) {
                    formData.append('stream', 'true');
                    const resp = await fetch('/api/transcribe', {
                        method: 'POST',
                        headers: { 'Accept': 'text/event-stream' },
                        body: formData,
                    });

                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        throw new Error(err.detail || 'Streaming transcription failed');
                    }

                    document.getElementById('transcribeEmptyState').style.display = 'none';
                    const resContainer = document.getElementById('transcribeResultContainer');
                    resContainer.style.display = 'flex';
                    const resultArea = document.getElementById('transcribeResultText');
                    resultArea.value = '';

                    const banner = document.getElementById('hopfieldCorrectionBanner');
                    const list = document.getElementById('hopfieldCorrectionList');
                    const segList = document.getElementById('transcribeSegmentsList');
                    segList.innerHTML = '';
                    list.innerHTML = '';
                    banner.style.display = 'none';

                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    let buffer = '';
                    const accumulatedSegments = [];
                    const allCorrections = [];

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const blocks = buffer.split('\n\n');
                        buffer = blocks.pop(); // keep remainder

                        for (const block of blocks) {
                            if (!block.trim()) continue;
                            let eventType = 'message';
                            let dataStr = '';
                            for (const line of block.split('\n')) {
                                if (line.startsWith('event:')) eventType = line.slice(6).trim();
                                if (line.startsWith('data:')) dataStr = line.slice(5).trim();
                            }
                            if (!dataStr) continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (eventType === 'metadata') {
                                    document.getElementById('transcribeMeta').style.display = 'flex';
                                    document.getElementById('transcribeLangBadge').textContent = (data.language || 'Detected').toUpperCase();
                                    document.getElementById('transcribeDurationBadge').textContent = (data.duration ? data.duration.toFixed(1) : '0.0') + 's';
                                } else if (eventType === 'segment') {
                                    accumulatedSegments.push(data.text);
                                    resultArea.value = accumulatedSegments.join(' ');
                                    resultArea.scrollTop = resultArea.scrollHeight;

                                    document.getElementById('transcribeSegmentsWrapper').style.display = 'flex';
                                    const item = document.createElement('div');
                                    item.style.cssText = 'display:flex; justify-content:space-between; align-items:center; background:#F8FAFC; border:1px solid var(--border-subtle); padding:6px 10px; border-radius:var(--radius-sm); font-size:12px;';
                                    item.innerHTML = `
                                        <span style="color:var(--text-main); font-weight:500;">${data.text}</span>
                                        <span style="color:var(--text-muted); font-family:monospace; font-size:11px;">${data.start.toFixed(2)}s – ${data.end.toFixed(2)}s</span>
                                    `;
                                    segList.appendChild(item);
                                } else if (eventType === 'correction') {
                                    allCorrections.push(data);
                                    banner.style.display = 'flex';
                                    const chip = document.createElement('span');
                                    chip.style.cssText = 'background:#DCFCE7; border:1px solid #86EFAC; padding:3px 10px; border-radius:var(--radius-full); font-weight:600; font-size:11px; display:inline-flex; align-items:center; gap:4px;';
                                    chip.innerHTML = `<span style="text-decoration:line-through; color:#991B1B;">${data.original}</span> → <span style="color:#166534; font-weight:800;">${data.corrected}</span> <span style="font-size:10px; color:#15803D; opacity:0.8;">(${(data.confidence * 100).toFixed(0)}%)</span>`;
                                    list.appendChild(chip);
                                } else if (eventType === 'complete') {
                                    if (data.text) resultArea.value = data.text;
                                    if (allCorrections.length > 0) {
                                        showToast(`✓ Hopfield Memory corrected ${allCorrections.length} word spelling(s)!`);
                                    } else {
                                        showToast('✓ Real-time streaming transcription completed!');
                                    }
                                }
                            } catch (parseErr) {
                                console.warn("SSE parse error", parseErr);
                            }
                        }
                    }
                    return;
                }

                // Standard non-streaming fallback
                const resp = await fetch('/api/transcribe', {
                    method: 'POST',
                    body: formData,
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Transcription failed');
                }

                const data = await resp.json();

                // Display result
                document.getElementById('transcribeEmptyState').style.display = 'none';
                const resContainer = document.getElementById('transcribeResultContainer');
                resContainer.style.display = 'flex';
                document.getElementById('transcribeResultText').value = data.text;

                // Meta badges
                document.getElementById('transcribeMeta').style.display = 'flex';
                document.getElementById('transcribeLangBadge').textContent = (data.language || 'Detected').toUpperCase();
                document.getElementById('transcribeDurationBadge').textContent = (data.duration ? data.duration.toFixed(1) : '0.0') + 's';

                // Hopfield correction banner
                const banner = document.getElementById('hopfieldCorrectionBanner');
                const list = document.getElementById('hopfieldCorrectionList');
                if (data.corrections && data.corrections.length > 0) {
                    banner.style.display = 'flex';
                    list.innerHTML = '';
                    data.corrections.forEach(c => {
                        const chip = document.createElement('span');
                        chip.style.cssText = 'background:#DCFCE7; border:1px solid #86EFAC; padding:3px 10px; border-radius:var(--radius-full); font-weight:600; font-size:11px; display:inline-flex; align-items:center; gap:4px;';
                        chip.innerHTML = `<span style="text-decoration:line-through; color:#991B1B;">${c.original}</span> → <span style="color:#166534; font-weight:800;">${c.corrected}</span> <span style="font-size:10px; color:#15803D; opacity:0.8;">(${(c.confidence * 100).toFixed(0)}%)</span>`;
                        list.appendChild(chip);
                    });
                    showToast(`✓ Hopfield Memory corrected ${data.corrections.length} word spelling(s)!`);
                } else {
                    banner.style.display = 'none';
                    list.innerHTML = '';
                    showToast('✓ Transcription completed successfully!');
                }

                // Render segments
                const segList = document.getElementById('transcribeSegmentsList');
                segList.innerHTML = '';
                if (data.segments && data.segments.length > 0) {
                    document.getElementById('transcribeSegmentsWrapper').style.display = 'flex';
                    data.segments.forEach(seg => {
                        const item = document.createElement('div');
                        item.style.cssText = 'display:flex; justify-content:space-between; align-items:center; background:#F8FAFC; border:1px solid var(--border-subtle); padding:6px 10px; border-radius:var(--radius-sm); font-size:12px;';
                        item.innerHTML = `
                            <span style="color:var(--text-main);">${seg.text}</span>
                            <span style="color:var(--text-muted); font-family:monospace; font-size:11px;">${seg.start.toFixed(2)}s – ${seg.end.toFixed(2)}s</span>
                        `;
                        segList.appendChild(item);
                    });
                } else {
                    document.getElementById('transcribeSegmentsWrapper').style.display = 'none';
                }
            } catch (err) {
                showToast(`Error: ${err.message}`);
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalBtnText;
            }
        }

        function copyTranscription() {
            const txt = document.getElementById('transcribeResultText').value;
            if (txt) {
                navigator.clipboard.writeText(txt);
                showToast('✓ Copied transcription to clipboard!');
            }
        }

        function sendToStudio() {
            const txt = document.getElementById('transcribeResultText').value;
            if (txt) {
                setPrompt(txt);
                switchTab('studio');
                showToast('✓ Text copied to Synthesis Studio!');
            }
        }

        function sendToLearner() {
            const txt = document.getElementById('transcribeResultText').value;
            if (txt) {
                document.getElementById('correctCarrier').value = txt;
                switchTab('learner');
                showToast('✓ Text copied to Pronunciation Learner!');
            }
        }

        // Initial system check
        fetch('/api/status').then(r => r.json()).then(s => {
            if (s.memory_entries !== undefined) {
                document.getElementById('memCountHeader').textContent = s.memory_entries;
                const memCountEl = document.getElementById('transcribeMemCount');
                if (memCountEl) memCountEl.textContent = s.memory_entries;
            }
        }).catch(()=>{});
        loadS3SpellingEntries();
    </script>
</body>
</html>
"""


def get_ui_html() -> str:
    """Return the raw HTML content string for the Web UI."""
    return HTML_CONTENT

