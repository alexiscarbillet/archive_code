import {
    useState,
    useRef
  } from "react"; 
  import "./App.css";
  
  function App() { 
    const inputRef = useRef(null); 
    const resultRef = useRef(null); 
    const [result, setResult] = useState(0); 
    
    // Addition function
    function plus(e) { 
      e.preventDefault(); 
      setResult((prevResult) => prevResult + Number(inputRef.current.value)); 
    };
  
    // Subtraction function
    function minus(e) { 
      e.preventDefault();
      setResult((prevResult) => prevResult - Number(inputRef.current.value)); 
    };
  
    // Multiplication function
    function times(e) { 
      e.preventDefault();
      setResult((prevResult) => prevResult * Number(inputRef.current.value)); 
    };
  
    // Division function
    function divide(e) { 
      e.preventDefault();
      const inputValue = Number(inputRef.current.value);
      if (inputValue === 0) {
        alert("Cannot divide by zero");
      } else {
        setResult((prevResult) => prevResult / inputValue);
      }
    };
  
    // Reset the input field
    function resetInput(e) { 
      e.preventDefault();
      inputRef.current.value = ""; 
    };
  
    // Reset the result value to 0
    function resetResult(e) { 
      e.preventDefault();
      setResult(0);
    };
  
    return ( 
      <div className="App"> 
        <div> 
          <h1>Simplest Working Calculator</h1> 
        </div> 
        <form> 
          <p ref={resultRef}> 
            {result}  {/* Display the current result */}
          </p> 
          <input
            pattern="[0-9]"
            ref={inputRef} 
            type="number" 
            placeholder="Type a number" 
          /> 
          <button onClick={plus}>Add</button> 
          <button onClick={minus}>Subtract</button> {/* Subtract button */}
          <button onClick={times}>Multiply</button> {/* Multiply button */}
          <button onClick={divide}>Divide</button> {/* Divide button */}
          <button onClick={resetInput}>Reset Input</button> {/* Reset Input button */}
          <button onClick={resetResult}>Reset Result</button> {/* Reset Result button */}
        </form> 
      </div> 
    ); 
  } 
  
   
  export default App; 
  