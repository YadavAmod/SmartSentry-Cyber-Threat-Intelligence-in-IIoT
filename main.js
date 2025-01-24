async function runModel(modelName) {
  const response = await fetch(`/run_model/${modelName}`);

  if (!response.ok) {
    document.getElementById("output").innerText = "Error running model!";
    return;
  }

  const data = await response.json();

  // Display the output in the output div
  document.getElementById("output").innerText = data.result; // Show results from the selected model
}
