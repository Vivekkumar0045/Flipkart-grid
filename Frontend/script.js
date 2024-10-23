async function fetchProducts() {
    try {
        const response = await fetch('http://127.0.0.1:8000/products');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch (error) {
        console.error('There was a problem fetching the products:', error);
        return [];
    }
}

async function loadProducts() {
    const loader = document.getElementById("loader");
    const table = document.getElementById("productTable");
    const tableBody = document.querySelector("#productTable tbody");

    const products = await fetchProducts();

    loader.style.display = "none";

    if (products.length > 0) {
        table.style.display = "table";

        products.forEach(product => {
            const row = document.createElement("tr");

            const nameCell = document.createElement("td");
            nameCell.textContent = product.productname;
            row.appendChild(nameCell);

            const quantityCell = document.createElement("td");
            quantityCell.textContent = product.quantity;
            row.appendChild(quantityCell);

            const categoryCell = document.createElement("td");
            categoryCell.textContent = product.category;
            row.appendChild(categoryCell);

            const detailCell = document.createElement("td");
            if (product.category === "FreshType-1" || product.category === "FreshType-2") {
                detailCell.textContent = product.freshness ? `${product.freshness}% Fresh` : 'N/A';
            } else if (product.category === "BottleType-1" || product.category === "PackType-1") {
                detailCell.textContent = product.expiry ? product.expiry : 'N/A';
            }
            row.appendChild(detailCell);

            tableBody.appendChild(row);
        });
    } else {
        const noDataMessage = document.createElement("tr");
        noDataMessage.innerHTML = '<td colspan="4">No products available.</td>';
        tableBody.appendChild(noDataMessage);
    }
}

document.addEventListener("DOMContentLoaded", loadProducts);
