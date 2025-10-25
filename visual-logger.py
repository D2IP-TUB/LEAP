import os
import json
import pandas as pd
import re

log_dir = "table_logs"
files = [f for f in os.listdir(log_dir) if f.endswith("_log.json")]
files.sort()

# Read the parallel_results.jsonl file
results_data = {}
try:
    with open("parallel_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                id_parts = data["id"].split("-")
                if len(id_parts) >= 2:
                    jsonl_num = int(id_parts[1])
                    log_num = jsonl_num - 1
                    request_id_base = f"req_{log_num}"
                    results_data[request_id_base] = {
                        "question": data["question"],
                        "ground_truth_answers": data["ground_truth_answers"],
                        "execution_accuracy": data["execution_accuracy"],
                        "actions": data["actions"]
                    }
    print(f"Loaded data for {len(results_data)} requests")
except FileNotFoundError:
    print("parallel_results.jsonl not found")
except Exception as e:
    print(f"Error reading parallel_results.jsonl: {e}")

processed_files = {}

for f in files:
    with open(os.path.join(log_dir, f), "r", encoding="utf-8") as infile:
        content = infile.read()
        entries = [e.strip() for e in content.split("-"*80) if e.strip()]
        json_entries = []
        
        for e in entries:
            entry_data = json.loads(e)
            full_request_id = entry_data['request_id']
            step = entry_data['step']
            
            base_request_id = None
            match = re.match(r'(req_\d+)', full_request_id)
            if match:
                base_request_id = match.group(1)
            
            if base_request_id and base_request_id in results_data:
                entry_data['question'] = results_data[base_request_id]['question']
                entry_data['ground_truth_answers'] = results_data[base_request_id]['ground_truth_answers']
                entry_data['execution_accuracy'] = results_data[base_request_id]['execution_accuracy']
                entry_data['expected_actions'] = results_data[base_request_id]['actions']
            
            csv_filename = None
            for csv_file in os.listdir(log_dir):
                if csv_file.startswith(full_request_id) and csv_file.endswith('.csv'):
                    if f"step{step:02d}" in csv_file:
                        csv_filename = csv_file
                        break
                    elif step < 10 and f"step{step}" in csv_file and f"step{step:02d}" not in csv_file:
                        if f"step{step}_" in csv_file or f"step{step}." in csv_file or csv_file.endswith(f"step{step}.csv"):
                            csv_filename = csv_file
                            break
            
            if csv_filename:
                try:
                    csv_path = os.path.join(log_dir, csv_filename)
                    df = pd.read_csv(csv_path)
                    entry_data['csv_data'] = {
                        'columns': df.columns.tolist(),
                        'rows': df.values.tolist()
                    }
                except Exception:
                    pass
            
            json_entries.append(entry_data)
        
        processed_files[f] = json_entries

# Create HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interactive Log Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .table-container {{ margin-top: 20px; max-height: 400px; overflow-y: auto; }}
    table, th, td {{
      border: 1px solid black;
      border-collapse: collapse;
      padding: 5px;
    }}
    th {{ background-color: #f0f0f0; position: sticky; top: 0; }}
    .success {{ background-color: #d4edda; padding: 10px; }}
    .failure {{ background-color: #f8d7da; padding: 10px; }}
    .step-header {{ 
        margin-top: 15px; 
        font-weight: bold; 
        font-size: 1.2em;
        padding: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
    }}
    .navigation {{ 
        padding: 15px;
        background-color: #e9ecef;
        border-radius: 5px;
    }}
    .question-info {{
        background-color: #fff3cd;
        border-radius: 4px;
        border-left: 3px solid #ffc107;
        padding: 20px;
    }}
    .question-info h3 {{
        margin-top: 0;
        color: black;
        font-size: 1.4em;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 10px;
    }}
    .info-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }}
    @media (max-width: 768px) {{
        .info-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    .metadata-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin: 15px 0;
        padding: 15px;
        background-color: #e9ecef;
        border-radius: 8px;
    }}
    .metadata-item {{
        display: flex;
        flex-direction: column;
    }}
    .metadata-label {{
        font-weight: bold;
        color: #495057;
        margin-bottom: 5px;
    }}
    .metadata-value {{
        font-size: 1.1em;
        color: #212529;
    }}
    .actions-section {{
        background-color: #fff3cd;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }}
    .action-item {{
        margin: 8px 0;
        padding: 8px;
        background-color: white;
        border-radius: 4px;
        border-left: 3px solid #007bff;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .action-item:hover {{
        background-color: #f8f9fa;
        transform: translateX(2px);
    }}
    .current-action {{
        background-color: #e7f3ff;
        border-left: 3px solid #28a745;
        font-weight: bold;
    }}
    .searchable-dropdown {{
        position: relative;
        display: inline-block;
        width: 250px;
        padding-right: 50px;
    }}
    .searchable-dropdown input {{
        width: 100%;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }}
    .dropdown-content {{
        display: none;
        position: absolute;
        background-color: #f9f9f9;
        min-width: 300px;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ccc;
        border-radius: 4px;
        z-index: 1;
    }}
    .dropdown-content.show {{
        display: block;
    }}
    .dropdown-option {{
        padding: 8px 12px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
    }}
    .dropdown-option:hover {{
        background-color: #e9ecef;
    }}
  </style>
</head>
<body>
  <h1>Interactive Log Viewer</h1>

  <div class="navigation">
    <label for="fileSearch">Search log file:</label>
    <div class="searchable-dropdown">
      <input type="text" id="fileSearch" placeholder="Type to search files..." onfocus="showDropdown()" oninput="filterFiles()">
      <div id="dropdownContent" class="dropdown-content"></div>
    </div>
    <button onclick="prevStep()">Previous Step</button>
    <button onclick="nextStep()">Next Step</button>
    <span id="stepCounter" style="margin-left: 20px; font-weight: bold;"></span>
  </div>

  <div id="questionContainer"></div>
  <div id="stepsContainer"></div>

  <script>
    const logs = {json.dumps(processed_files)};
    const files = Object.keys(logs);
    let currentData = [];
    let currentStep = 0;
    let maxStep = 0;
    let currentFilename = '';

    function showDropdown() {{
      const dropdown = document.getElementById('dropdownContent');
      dropdown.innerHTML = '';
      files.forEach(f => {{
        const option = document.createElement('div');
        option.className = 'dropdown-option';
        option.textContent = f;
        option.onclick = () => selectFile(f);
        dropdown.appendChild(option);
      }});
      dropdown.classList.add('show');
    }}

    function filterFiles() {{
      const input = document.getElementById('fileSearch');
      const filter = input.value.toLowerCase();
      const dropdown = document.getElementById('dropdownContent');
      const options = dropdown.getElementsByClassName('dropdown-option');
      
      for (let i = 0; i < options.length; i++) {{
        const text = options[i].textContent.toLowerCase();
        if (text.includes(filter)) {{
          options[i].style.display = '';
        }} else {{
          options[i].style.display = 'none';
        }}
      }}
    }}

    function selectFile(filename) {{
      document.getElementById('fileSearch').value = filename;
      document.getElementById('dropdownContent').classList.remove('show');
      currentFilename = filename;
      loadFile(filename);
    }}

    document.addEventListener('click', function(event) {{
      const dropdown = document.getElementById('dropdownContent');
      const searchInput = document.getElementById('fileSearch');
      if (!searchInput.contains(event.target) && !dropdown.contains(event.target)) {{
        dropdown.classList.remove('show');
      }}
    }});

    function loadFile(filename) {{
      currentData = logs[filename];
      currentStep = 0;
      
      maxStep = currentData.length - 1;
      for (let i = 0; i < currentData.length; i++) {{
        if (currentData[i].failure_type) {{
          maxStep = i;
          break;
        }}
      }}
      
      renderStep();
    }}

    function parseTablePreview(tablePreview) {{
      if (!tablePreview || !Array.isArray(tablePreview)) {{
        return {{ columns: [], rows: [] }};
      }}
      
      try {{
        const fullText = tablePreview.join('');
        const lines = fullText.split('\\r\\n').filter(line => line.trim());
        
        if (lines.length === 0) {{
          return {{ columns: [], rows: [] }};
        }}
        
        const columns = lines[0].split(',').map(col => col.trim());
        const rows = lines.slice(1).map(line => {{
          return line.split(',').map(cell => cell.trim());
        }});
        
        return {{ columns, rows }};
      }} catch (e) {{
        return {{ columns: [], rows: [] }};
      }}
    }}

    function jumpToStep(stepNumber) {{
      if (stepNumber >= 0 && stepNumber <= maxStep) {{
        currentStep = stepNumber;
        renderStep();
      }}
    }}

    function renderStep() {{
      if (currentData.length === 0) return;
      const stepData = currentData[currentStep];

      const container = document.getElementById("stepsContainer");
      const questionContainer = document.getElementById("questionContainer");
      container.innerHTML = "";
      
      if (stepData.question) {{
        questionContainer.innerHTML = "";
        
        const infoGrid = document.createElement("div");
        infoGrid.className = "info-grid";

        const questionDiv = document.createElement("div");
        questionDiv.className = "question-info";
        questionDiv.innerHTML = 
          '<h3>Question</h3>' + 
          '<p style="font-size: 1.2em; line-height: 1.5;">' + stepData.question + '</p>';
        infoGrid.appendChild(questionDiv);

        if (stepData.ground_truth_answers && stepData.ground_truth_answers.length > 0) {{
          const answerDiv = document.createElement("div");
          answerDiv.className = "question-info";
          answerDiv.innerHTML = 
            '<h3>Answer</h3>' + 
            '<p style="font-size: 1.2em; line-height: 1.5;">' + stepData.ground_truth_answers.join(', ') + '</p>';
          infoGrid.appendChild(answerDiv);
        }}

        questionContainer.appendChild(infoGrid);

        const metadataGrid = document.createElement("div");
        metadataGrid.className = "metadata-grid";

        if (stepData.execution_accuracy !== undefined) {{
          const accuracyDiv = document.createElement("div");
          accuracyDiv.className = "metadata-item";
          accuracyDiv.innerHTML = 
            '<div class="metadata-label">Execution Accuracy</div>' +
            '<div class="metadata-value" style="color: ' + (stepData.execution_accuracy === 1 ? '#28a745' : '#dc3545') + '; font-weight: bold;">' +
            stepData.execution_accuracy +
            '</div>';
          metadataGrid.appendChild(accuracyDiv);
        }}

        if (stepData.generation_mode) {{
          const generationDiv = document.createElement("div");
          generationDiv.className = "metadata-item";
          generationDiv.innerHTML = 
            '<div class="metadata-label">Generation Mode</div>' +
            '<div class="metadata-value">' + stepData.generation_mode + '</div>';
          metadataGrid.appendChild(generationDiv);
        }}

        questionContainer.appendChild(metadataGrid);

        // Create complete action sequence including initial step
        const allActions = [];
        
        // Add initial step
        allActions.push({{ action: 'initial', args: [] }});
        
        // Add expected actions from JSONL
        if (stepData.expected_actions && stepData.expected_actions.length > 0) {{
          stepData.expected_actions.forEach(action => {{
            allActions.push(action);
          }});
        }}

        // Display all actions as clickable items
        if (allActions.length > 0) {{
          const actionsSection = document.createElement("div");
          actionsSection.className = "actions-section";
          actionsSection.innerHTML = '<h3 style="margin-top: 0;">All Actions</h3>';
          
          allActions.forEach((action, index) => {{
            const actionDiv = document.createElement("div");
            actionDiv.className = "action-item";
            if (index === currentStep) {{
              actionDiv.classList.add("current-action");
            }}
            
            const actionText = action.action + '(' + (action.args ? action.args.map(arg => JSON.stringify(arg)).join(', ') : '') + ')';
            actionDiv.innerHTML = '<strong>Step ' + index + ':</strong> ' + actionText;
            actionDiv.onclick = () => jumpToStep(index);
            actionsSection.appendChild(actionDiv);
          }});
          
          questionContainer.appendChild(actionsSection);
        }}
      }}

      const header = document.createElement("div");
      header.className = "step-header";
      let headerText = 'Step ' + currentStep + ' of ' + maxStep + ' - Request ID: ' + stepData.request_id;
      if (stepData.failure_type) {{
        headerText += ' | Failure: ' + stepData.failure_type;
      }}
      header.textContent = headerText;
      container.appendChild(header);

      const status = document.createElement("div");
      status.textContent = 'Action: ' + stepData.action + ' | Success: ' + stepData.success;
      
      if (stepData.execution_accuracy !== undefined) {{
        status.className = stepData.execution_accuracy === 1 ? "success" : "failure";
        status.textContent += ' | Execution Accuracy: ' + stepData.execution_accuracy;
      }} else {{
        status.className = stepData.success ? "success" : "failure";
      }}
      container.appendChild(status);

      let tableData = null;
      if (stepData.csv_data && stepData.csv_data.columns && stepData.csv_data.columns.length > 0) {{
        tableData = stepData.csv_data;
      }} else {{
        tableData = parseTablePreview(stepData.table_preview);
      }}

      const tableContainer = document.createElement("div");
      tableContainer.className = "table-container";

      if (tableData && tableData.columns && tableData.columns.length > 0) {{
        const table = document.createElement("table");
        
        const headerRow = document.createElement("tr");
        tableData.columns.forEach(col => {{
          const th = document.createElement("th");
          th.textContent = col;
          headerRow.appendChild(th);
        }});
        table.appendChild(headerRow);

        for (let i = 0; i < tableData.rows.length; i++) {{
          const tr = document.createElement("tr");
          tableData.columns.forEach((_, ci) => {{
            const td = document.createElement("td");
            const cellValue = tableData.rows[i] ? tableData.rows[i][ci] : '';
            td.textContent = cellValue !== undefined && cellValue !== null ? cellValue.toString() : "";
            tr.appendChild(td);
          }});
          table.appendChild(tr);
        }}
        tableContainer.appendChild(table);
      }} else {{
        const noData = document.createElement("div");
        noData.textContent = "No table data available";
        noData.style.padding = "10px";
        noData.style.backgroundColor = "#f8f9fa";
        tableContainer.appendChild(noData);
      }}
      container.appendChild(tableContainer);

      document.getElementById("stepCounter").textContent = 'Step ' + currentStep + ' of ' + maxStep;
    }}

    function prevStep() {{
      if (currentStep > 0) {{
        currentStep--;
        renderStep();
      }}
    }}

    function nextStep() {{
      if (currentStep < maxStep) {{
        currentStep++;
        renderStep();
      }}
    }}

    if (files.length > 0) {{
      currentFilename = files[0];
      document.getElementById('fileSearch').value = files[0];
      loadFile(files[0]);
    }}
  </script>
</body>
</html>
"""

with open("viewer.html", "w", encoding="utf-8") as f:
    f.write(html_content)
