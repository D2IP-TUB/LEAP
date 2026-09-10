document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('experimentForm');
    const submitBtn = document.getElementById('submitBtn');
    const loader = document.getElementById('loader');
    const resultsArea = document.getElementById('resultsArea');
    const logsContainer = document.getElementById('logsContainer');
    const errorArea = document.getElementById('errorArea');
    const actionListDiv = document.getElementById('actionList');

    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const stepIndicator = document.getElementById('stepIndicator');

    let currentSteps = [];
    let currentStepIndex = 0;
    
    let currentTablePage = 1;
    const ROWS_PER_PAGE = 100;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resetUI();

        const formData = new FormData(form);

        try {
            const response = await fetch('/api/run', { method: 'POST', body: formData });
            const data = await response.json();

            const stepsData = Array.isArray(data) ? data : data.steps;
            if (!stepsData || stepsData.length === 0) throw new Error("No steps returned.");

            currentSteps = stepsData;
            currentStepIndex = 0;

            renderActionList();
            renderCurrentStep();
            
            resultsArea.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            errorArea.textContent = error.message;
            errorArea.classList.remove('hidden');
        } finally {
            loader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStepIndex > 0) {
            currentStepIndex--;
            renderCurrentStep();
            updateActionListHighlight();
        }
    });

    nextBtn.addEventListener('click', () => {
        if (currentStepIndex < currentSteps.length - 1) {
            currentStepIndex++;
            renderCurrentStep();
            updateActionListHighlight();
        }
    });


    function renderActionList() {
        actionListDiv.innerHTML = '';
        currentSteps.forEach((step, idx) => {
            const card = document.createElement('div');
            card.className = 'action-card';
            if (idx === currentStepIndex) card.classList.add('active');
            
            const name = getActionName(step);
            card.textContent = `Step ${idx}: ${name}`;
            
            card.addEventListener('click', () => {
                currentStepIndex = idx;
                renderCurrentStep();
                updateActionListHighlight();
            });

            actionListDiv.appendChild(card);
        });
    }

    function updateActionListHighlight() {
        const cards = actionListDiv.querySelectorAll('.action-card');
        cards.forEach((card, idx) => {
            if (idx === currentStepIndex) card.classList.add('active');
            else card.classList.remove('active');
        });
    }

    function renderCurrentStep() {
        const step = currentSteps[currentStepIndex];
        
        stepIndicator.textContent = `Step ${currentStepIndex} of ${currentSteps.length - 1}`;
        prevBtn.disabled = (currentStepIndex === 0);
        nextBtn.disabled = (currentStepIndex === currentSteps.length - 1);

        logsContainer.innerHTML = '';
        
        currentTablePage = 1;

        const actionName = getActionName(step);
        const actionArgs = getActionArgs(step);

        const actionTitle = document.createElement('div');
        actionTitle.className = 'action-main-title';
        actionTitle.textContent = `Action: ${actionName} ${actionArgs}`;
        logsContainer.appendChild(actionTitle);

        const metaBox = document.createElement('div');
        metaBox.className = 'step-metadata-box';
        metaBox.textContent = `Step ${step.index} of ${currentSteps.length - 1}`;
        logsContainer.appendChild(metaBox);

        if (step.sampling_metadata) {
            const md = step.sampling_metadata;
            const candidatesDiv = document.createElement('div');
            candidatesDiv.className = 'candidates-container';
            
            const statsText = `Requested: ${md.n_requested || 8}, Generated: ${md.n_generated || 8}, Valid: ${md.valid_actions ? md.valid_actions.length : 0}`;
            
            let listHtml = '<ol class="candidate-list-ol">';
            if (md.candidate_actions && Array.isArray(md.candidate_actions)) {
                md.candidate_actions.forEach(cand => {
                    const isWinner = cand.includes(actionName) && (step.index > 0); 
                    const winnerClass = isWinner ? 'winner-row' : '';
                    listHtml += `<li class="candidate-li ${winnerClass}">${cand}</li>`;
                });
            }
            listHtml += '</ol>';

            candidatesDiv.innerHTML = `
                <div class="candidates-header">Sampling Candidates (this step)</div>
                <div class="stats-bar">${statsText}</div>
                ${listHtml}
            `;
            logsContainer.appendChild(candidatesDiv);
        }

        const tableContainer = document.createElement('div');
        logsContainer.appendChild(tableContainer);

        renderTableWithPagination(step.table, tableContainer);
    }


    function renderTableWithPagination(tableData, container) {
        container.innerHTML = ''; 

        if (!tableData || !tableData.columns || !tableData.rows || tableData.rows.length === 0) {
            container.innerHTML = '<div style="padding:20px; text-align:center;"><em>Empty Table</em></div>';
            return;
        }

        const totalRows = tableData.rows.length;
        const totalPages = Math.ceil(totalRows / ROWS_PER_PAGE);

        const startIndex = (currentTablePage - 1) * ROWS_PER_PAGE;
        const endIndex = Math.min(startIndex + ROWS_PER_PAGE, totalRows);
        const currentRows = tableData.rows.slice(startIndex, endIndex);

        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'table-wrapper';

        const columns = tableData.columns;
        let html = '<table class="result-table"><thead><tr>';
        columns.forEach(col => html += `<th>${col}</th>`);
        html += '</tr></thead><tbody>';
        
        currentRows.forEach(row => {
            html += '<tr>';
            row.forEach(cell => html += `<td>${cell !== null ? cell : ''}</td>`);
            html += '</tr>';
        });
        html += '</tbody></table>';
        
        tableWrapper.innerHTML = html;
        container.appendChild(tableWrapper);

        if (totalRows > ROWS_PER_PAGE) {
            const controls = document.createElement('div');
            controls.className = 'table-pagination-controls';
            
            const btnPrev = document.createElement('button');
            btnPrev.className = 'table-page-btn';
            btnPrev.textContent = '← Previous';
            btnPrev.disabled = currentTablePage === 1;
            btnPrev.onclick = () => {
                currentTablePage--;
                renderTableWithPagination(tableData, container);
            };

            const info = document.createElement('span');
            info.className = 'pagination-info-text';
            info.textContent = `${startIndex + 1}-${endIndex} of ${totalRows}`;


            const btnNext = document.createElement('button');
            btnNext.className = 'table-page-btn';
            btnNext.textContent = 'Next →';
            btnNext.disabled = currentTablePage === totalPages;
            btnNext.onclick = () => {
                currentTablePage++;
                renderTableWithPagination(tableData, container);
            };

            controls.appendChild(btnPrev);
            controls.appendChild(info);
            controls.appendChild(btnNext);
            
            container.appendChild(controls);
        }
    }

    function getActionName(step) {
        if (typeof step.action === 'string') return step.action;
        if (step.action && step.action.name) return step.action.name;
        return "Unknown";
    }

    function getActionArgs(step) {
        if (step.action && step.action.arguments && Array.isArray(step.action.arguments)) {
             const formattedArgs = step.action.arguments.map(arg => 
                typeof arg === 'string' ? `'${arg}'` : arg
            ).join(', ');
            return `(${formattedArgs})`;
        }
        return "()";
    }

    function resetUI() {
        resultsArea.classList.add('hidden');
        errorArea.classList.add('hidden');
        logsContainer.innerHTML = '';
        actionListDiv.innerHTML = '';
        loader.classList.remove('hidden');
        submitBtn.disabled = true;
    }
});