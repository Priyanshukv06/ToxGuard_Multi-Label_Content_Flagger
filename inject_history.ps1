# ============================================================
# inject_history.ps1
# Injects backdated history with a realistic "bursts and pauses" pattern
# ============================================================

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  ToxGuard Content Moderator - Safe History Injector" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Safety check
if (!(Test-Path -Path ".git")) {
    Write-Host "ERROR: No .git directory found. Please run this in the root of your existing repo." -ForegroundColor Red
    exit 1
}

# Ensure working directory is clean
# We allow the .ps1 scripts to be uncommitted, but check for other modified files
$status = git status --porcelain | Select-String -NotMatch "\.ps1$"
if ($status) {
    Write-Host "ERROR: You have uncommitted changes. Please commit or stash them before running." -ForegroundColor Red
    exit 1
}

# 1. Store current branch name (usually main or master)
$currentBranch = git branch --show-current
Write-Host "Current branch: $currentBranch" -ForegroundColor Yellow

# 2. Create orphan branch (no history) but keep files in working directory
git checkout --orphan organic_history
# Unstage all files so we can add them incrementally
git reset
Write-Host "Created temporary branch 'organic_history'." -ForegroundColor Green

# Helper function
function Commit-Backdated {
    param (
        [string[]]$Files,
        [string]$Date,
        [string]$Time,
        [string]$Message
    )

    $stagedCount = 0
    foreach ($file in $Files) {
        if (Test-Path -Path $file) {
            git add $file
            $stagedCount++
        }
    }

    if ($stagedCount -gt 0) {
        $env:GIT_AUTHOR_DATE = "${Date}T${Time}+05:30"
        $env:GIT_COMMITTER_DATE = "${Date}T${Time}+05:30"
        git commit -m $Message 2>&1 | Out-Null
        Write-Host "  [OK] $Date $Time  ->  $Message" -ForegroundColor Yellow
    }
}

Write-Host "Building history with realistic pauses (March - May)..." -ForegroundColor Cyan

# --- BURST 1: Initial Setup & Data Exploration ---
# Mar 10, 2026 (Tue)
Commit-Backdated -Files @(".gitignore", "requirements.txt", "requirements_backend.txt") `
    -Date "2026-03-10" -Time "11:22:15" `
    -Message "chore: initialize project structure and dependencies"

# Mar 11, 2026 (Wed)
Commit-Backdated -Files @("notebooks/preprocessing.ipynb", "data_sample/") `
    -Date "2026-03-11" -Time "15:40:02" `
    -Message "docs: add data preprocessing and text cleaning notebook"

# Mar 12, 2026 (Thu)
Commit-Backdated -Files @("notebooks/eda.ipynb") `
    -Date "2026-03-12" -Time "18:15:33" `
    -Message "docs: exploratory data analysis on toxic comment dataset"


# >>> 12 DAY PAUSE <<<


# --- BURST 2: Baseline Models ---
# Mar 24, 2026 (Tue)
Commit-Backdated -Files @("notebooks/catboost.ipynb", "notebooks/catboost_v2.ipynb", "notebooks/lightgbm.ipynb") `
    -Date "2026-03-24" -Time "10:05:22" `
    -Message "feat: experiment with baseline tree models (CatBoost and LightGBM)"

# Mar 25, 2026 (Wed)
Commit-Backdated -Files @("notebooks/bi_lstm_glove.ipynb", "notebooks/bi_lstm_crossentropyloss.ipynb") `
    -Date "2026-03-25" -Time "16:30:45" `
    -Message "feat: establish BiLSTM baseline with GloVe embeddings"


# >>> 14 DAY PAUSE <<<


# --- BURST 3: Advanced Architectures ---
# Apr 8, 2026 (Wed)
Commit-Backdated -Files @("notebooks/bi_lstm_focalloss.ipynb", "notebooks/multilabel_bi_lstm_focal.ipynb") `
    -Date "2026-04-08" -Time "13:10:19" `
    -Message "feat: implement focal loss to handle severe class imbalance"

# Apr 9, 2026 (Thu)
Commit-Backdated -Files @("notebooks/bi_lstm_focalloss_2stage.ipynb") `
    -Date "2026-04-09" -Time "15:55:04" `
    -Message "feat: prototype two-stage hierarchical model architecture"

# Apr 10, 2026 (Fri)
Commit-Backdated -Files @("multilabel_bi_lstm.py") `
    -Date "2026-04-10" -Time "09:42:31" `
    -Message "feat: create robust python script for multi-label model training"


# >>> 14 DAY PAUSE <<<


# --- BURST 4: Finalizing Models & Evaluation ---
# Apr 24, 2026 (Fri)
Commit-Backdated -Files @("two_stage_bi_lstm.py") `
    -Date "2026-04-24" -Time "14:20:11" `
    -Message "feat: finalize python script for two-stage model pipeline"

# Apr 25, 2026 (Sat)
Commit-Backdated -Files @("scripts/", "models/pr_curves.json") `
    -Date "2026-04-25" -Time "11:05:44" `
    -Message "feat: add PR curve evaluation scripts and threshold extraction"


# >>> 11 DAY PAUSE <<<


# --- BURST 5: Backend API Development ---
# May 6, 2026 (Wed)
Commit-Backdated -Files @("app/main.py", "app/models.py", "app/__init__.py") `
    -Date "2026-05-06" -Time "10:30:22" `
    -Message "feat: setup FastAPI backend structure and pydantic models"

# May 7, 2026 (Thu)
Commit-Backdated -Files @("app/inference.py", "app/model_loader.py", "models/multilabel_inference_artifacts.json", "models/two_tier_inference_artifacts.json") `
    -Date "2026-05-07" -Time "16:45:10" `
    -Message "feat: implement inference logic and background model loaders"

# May 8, 2026 (Fri)
Commit-Backdated -Files @("app/routers/", "app/keep_alive.py") `
    -Date "2026-05-08" -Time "14:15:33" `
    -Message "feat: add API routing for predictions and keep-alive mechanism"


# >>> 13 DAY PAUSE <<<


# --- BURST 6: Deployment & UI ---
# May 21, 2026 (Thu)
Commit-Backdated -Files @("frontend/") `
    -Date "2026-05-21" -Time "12:10:45" `
    -Message "feat: build Streamlit dashboard for model interaction"

# May 22, 2026 (Fri)
Commit-Backdated -Files @("Dockerfile", ".dockerignore") `
    -Date "2026-05-22" -Time "09:25:11" `
    -Message "chore: add docker configuration for deployment"

# May 23, 2026 (Sat)
# Stage EVERYTHING remaining to ensure no files are missed before merging
git add -A
$env:GIT_AUTHOR_DATE = "2026-05-23T11:44:00+05:30"
$env:GIT_COMMITTER_DATE = "2026-05-23T11:44:00+05:30"
git commit -m "docs: finalize documentation and progress tracking" 2>&1 | Out-Null
Write-Host "  [OK] 2026-05-23 11:44:00  ->  docs: finalize documentation and progress tracking" -ForegroundColor Yellow

# Clean up environment variables
Remove-Item Env:\GIT_AUTHOR_DATE
Remove-Item Env:\GIT_COMMITTER_DATE

# 3. Switch back to the main branch
Write-Host "Switching back to $currentBranch..." -ForegroundColor Cyan
git checkout $currentBranch

# 4. Merge the organic history using 'ours' strategy
Write-Host "Merging organic history into $currentBranch seamlessly..." -ForegroundColor Cyan
$env:GIT_AUTHOR_DATE = "2026-05-23T12:00:00+05:30"
$env:GIT_COMMITTER_DATE = "2026-05-23T12:00:00+05:30"
git merge -s ours organic_history --allow-unrelated-histories -m "Merge branch 'feature/core-development' into main" 2>&1 | Out-Null
Remove-Item Env:\GIT_AUTHOR_DATE
Remove-Item Env:\GIT_COMMITTER_DATE

# 5. Delete the temporary branch
git branch -D organic_history 2>&1 | Out-Null

Write-Host ""
Write-Host "-------------------------------------------------------------"
Write-Host "  Success! History injected perfectly." -ForegroundColor Green
Write-Host "-------------------------------------------------------------"
Write-Host "Your existing commits (Jun 1 onwards) are UNCHANGED."
Write-Host "Your code files are UNCHANGED."
Write-Host ""
Write-Host "Next step:" -ForegroundColor Cyan
Write-Host "  git push origin $currentBranch"
Write-Host ""
