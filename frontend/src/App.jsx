import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [customers, setCustomers] = useState([]);

  useEffect(() => {
    axios
      .get("http://localhost:8001/customers")
      .then((response) => {
        setCustomers(response.data.customers);
      })
      .catch((error) => {
        console.error("Error fetching customers:", error);
      });
  }, []);

  return (
    <ul>
      {customers.map((customer, index) => (
        <li key={index}>
          {index + 1}: {customer.first_name} {customer.last_name}
        </li>
      ))}
    </ul>
  );
}

export default App;