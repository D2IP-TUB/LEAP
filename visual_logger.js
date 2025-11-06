let logs = {};
let files = [];
let currentData = [];
let currentStep = 0;
let maxStep = 0;
let currentFilename = '';


function showDropdown() {
  const dropdown = document.getElementById('dropdownContent');
  dropdown.innerHTML = '';
  files.forEach(f => {
    const option = document.createElement('div');
    option.className = 'dropdown-option';
    option.textContent = f;
    option.onclick = () => selectFile(f);
    dropdown.appendChild(option);
  });
  dropdown.classList.add('show');
}

function filterFiles() {
  const input = document.getElementById('fileSearch');
  const filter = input.value.toLowerCase();
  const dropdown = document.getElementById('dropdownContent');
  const options = dropdown.getElementsByClassName('dropdown-option');
  
  for (let i = 0; i < options.length; i++) {
    const text = options[i].textContent.toLowerCase();
    if (text.includes(filter)) {
      options[i].style.display = '';
    } else {
      options[i].style.display = 'none';
    }
  }
}

function selectFile(filename) {
  document.getElementById('fileSearch').value = filename;
  document.getElementById('dropdownContent').classList.remove('show');
  currentFilename = filename;
  loadFile(filename);
}

document.addEventListener('click', function(event) {
  const dropdown = document.getElementById('dropdownContent');
  const searchInput = document.getElementById('fileSearch');
  if (searchInput && dropdown && !searchInput.contains(event.target) && !dropdown.contains(event.target)) {
    dropdown.classList.remove('show');
  }
});

function loadFile(filename) {
  currentData = logs[filename];
  currentStep = 0;
  
  maxStep = currentData.length - 1;
  for (let i = 0; i < currentData.length; i++) {
    if (currentData[i].failure_type) {
      maxStep = i;
      break;
    }
  }
  
  renderStep();
}

function parseTablePreview(tablePreview) {
  if (!tablePreview || !Array.isArray(tablePreview)) {
    return { columns: [], rows: [] };
  }
  
  try {
    const fullText = tablePreview.join('');
    const lines = fullText.split('\r\n').filter(line => line.trim());
    
    if (lines.length === 0) {
      return { columns: [], rows: [] };
    }
    
    const columns = lines[0].split(',').map(col => col.trim());
    const rows = lines.slice(1).map(line => {
      return line.split(',').map(cell => cell.trim());
    });
    
    return { columns, rows };
  } catch (e) {
    return { columns: [], rows: [] };
  }
}

function jumpToStep(stepNumber) {
  if (stepNumber >= 0 && stepNumber <= maxStep) {
    currentStep = stepNumber;
    renderStep();
  }
}
function renderStep() {
    if (currentData.length === 0) return;
  
    const stepData = currentData[currentStep]; 
    
    const questionStep = currentData.find(step => step.question);
  
    const container = document.getElementById("stepsContainer");
    const questionContainer = document.getElementById("questionContainer");
    

    container.innerHTML = "";
    questionContainer.innerHTML = ""; 

    if (questionStep) {
      const infoGrid = document.createElement("div");
      infoGrid.className = "info-grid";
      const questionDiv = document.createElement("div");
      questionDiv.className = "question-info";
      questionDiv.innerHTML = 
        '<h3>Question</h3>' + 
        '<p style="font-size: 1.2em; line-height: 1.5;">' + questionStep.question + '</p>';
      infoGrid.appendChild(questionDiv);
      if (questionStep.ground_truth_answers && questionStep.ground_truth_answers.length > 0) {
        const answerDiv = document.createElement("div");
        answerDiv.className = "question-info";
        answerDiv.innerHTML = 
          '<h3>Answer</h3>' + 
          '<p style="font-size: 1.2em; line-height: 1.5;">' + questionStep.ground_truth_answers.join(', ') + '</p>';
        infoGrid.appendChild(answerDiv);
      }
      questionContainer.appendChild(infoGrid);
      const metadataGrid = document.createElement("div");
      metadataGrid.className = "metadata-grid";
      if (questionStep.execution_accuracy !== undefined) {
        const accuracyDiv = document.createElement("div");
        accuracyDiv.className = "metadata-item";
        accuracyDiv.innerHTML = 
          '<div class="metadata-label">Execution Accuracy</div>' +
          '<div class="metadata-value" style="color: ' + (questionStep.execution_accuracy === 1 ? '#28a745' : '#dc3545') + '; font-weight: bold;">' +
          questionStep.execution_accuracy +
          '</div>';
        metadataGrid.appendChild(accuracyDiv);
      }
      if (questionStep.generation_mode) {
        const generationDiv = document.createElement("div");
        generationDiv.className = "metadata-item";
        generationDiv.innerHTML = 
          '<div class="metadata-label">Generation Mode</div>' +
          '<div class="metadata-value">' + questionStep.generation_mode + '</div>';
        metadataGrid.appendChild(generationDiv);
      }
      questionContainer.appendChild(metadataGrid);
  

      if (maxStep >= 0 && maxStep < currentData.length) {
        const actionsSection = document.createElement("div");
        actionsSection.className = "actions-section";
        actionsSection.innerHTML = '<h3 style="margin-top: 0;">All Actions</h3>';
        
        for (let i = 0; i <= maxStep; i++) {
          const actionData = currentData[i];
          const action = actionData.action || 'initial'; 
          const args = actionData.args || []; 
          
          const actionDiv = document.createElement("div");
          actionDiv.className = "action-item";
          
          if (i === currentStep) { 
            actionDiv.classList.add("current-action");
          }
          
          const actionText = action  + (args.length > 0 ? args.map(arg => JSON.stringify(arg)).join(', ') : '') ;
          actionDiv.innerHTML = '<strong>Step ' + i + ':</strong> ' + actionText;
          
          actionDiv.onclick = (function(stepNum) {
            return function() {
              jumpToStep(stepNum);
            };
          })(i);
          
          actionsSection.appendChild(actionDiv);
        }
        
        questionContainer.appendChild(actionsSection);
      }
    } 
    
    const header = document.createElement("div");
    header.className = "step-header";
    let headerText = 'Step ' + currentStep + ' of ' + maxStep + ' - Request ID: ' + stepData.request_id;
    if (stepData.failure_type) {
      headerText += ' | Failure: ' + stepData.failure_type;
    }
    header.textContent = headerText;
    container.appendChild(header);
  
    const status = document.createElement("div");
    status.textContent = 'Action: ' + stepData.action + ' | Success: ' + stepData.success;
    
    if (stepData.execution_accuracy !== undefined) {
      status.className = stepData.execution_accuracy === 1 ? "success" : "failure";
      status.textContent += ' | Execution Accuracy: ' + stepData.execution_accuracy;
    } else {
      status.className = stepData.success ? "success" : "failure";
    }
    container.appendChild(status);
  
    let tableData = null;
    if (stepData.csv_data && stepData.csv_data.columns && stepData.csv_data.columns.length > 0) {
      tableData = stepData.csv_data;
    } else {
      tableData = parseTablePreview(stepData.table_preview);
    }
  
    const tableContainer = document.createElement("div");
    tableContainer.className = "table-container";
  
    if (tableData && tableData.columns && tableData.columns.length > 0) {
      const table = document.createElement("table");
      
      const headerRow = document.createElement("tr");
      tableData.columns.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col;
        headerRow.appendChild(th);
      });
      table.appendChild(headerRow);
  
      for (let i = 0; i < tableData.rows.length; i++) {
        const tr = document.createElement("tr");
        tableData.columns.forEach((_, ci) => {
          const td = document.createElement("td");
          const cellValue = tableData.rows[i] ? tableData.rows[i][ci] : '';
          td.textContent = cellValue !== undefined && cellValue !== null ? cellValue.toString() : "";
          tr.appendChild(td);
        });
        table.appendChild(tr);
      }
      tableContainer.appendChild(table);
    } else {
      const noData = document.createElement("div");
      noData.textContent = "No table data available";
      noData.style.padding = "10px";
      noData.style.backgroundColor = "#f8f9fa";
      tableContainer.appendChild(noData);
    }
    container.appendChild(tableContainer);
  
    document.getElementById("stepCounter").textContent = 'Step ' + currentStep + ' of ' + maxStep;
  }
function prevStep() {
  if (currentStep > 0) {
    currentStep--;
    renderStep();
  }
}

function nextStep() {
  if (currentStep < maxStep) {
    currentStep++;
    renderStep();
  }
}
function jsonReviver(key, value) {
    if (typeof value === 'string') {
      if (value === "__NaN__") {
        return NaN;
      }
      if (value === "__Infinity__") {
        return Infinity;
      }
      if (value === "__-Infinity__") {
        return -Infinity;
      }
    }
    return value;
  }

fetch('data.json')
.then(response => {
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.text();
})
.then(text => {
  const data = JSON.parse(text, jsonReviver);
  
  logs = data;
  files = Object.keys(logs);
  
  if (files.length > 0) {
    currentFilename = files[0];
    document.getElementById('fileSearch').value = files[0];
    loadFile(files[0]);
  }
})
.catch(error => {
  console.error("Could not load or parse log data:", error);
  document.body.innerHTML = '<h1>Error</h1><p>Could not load <code>data.json</code>. Please check the file exists and the console for more details.</p>';
});