// script.js

document.addEventListener('DOMContentLoaded', () => {
    console.log('DataVision cargado correctamente.');

    const tocContainer = document.getElementById("toc-container");
    const toggleButton = document.getElementById("toggle-toc");
    const headers = document.querySelectorAll("main h1, main h2, main h3");
    let isManuallyHidden = false;

    // Mostrar la tabla cuando el cursor está en el margen izquierdo
    document.addEventListener("mousemove", (event) => {
        if (isManuallyHidden) return; // No hacer nada si está oculto manualmente

        // Detectar si el cursor está en el margen izquierdo
        if (event.clientX < 30) {
            tocContainer.classList.add("active");
        } else if (event.clientX > 260) { // Asegura que el cursor se aleja lo suficiente
            tocContainer.classList.remove("active");
        }
    });

    // Alternar visibilidad manualmente con el botón
    toggleButton.addEventListener("click", () => {
        isManuallyHidden = !tocContainer.classList.contains("active");
        tocContainer.classList.toggle("active");

        // Cambiar texto del botón
        toggleButton.textContent = tocContainer.classList.contains("active")
            ? "Ocultar"
            : "Mostrar";
    });

    // Crear un contenedor de lista
    const ul = document.createElement("ul");

    headers.forEach((header, index) => {
        const id = `header-${index}`;
        header.id = id; // Añade un ID único a cada encabezado

        // Crear elemento de la lista
        const li = document.createElement("li");
        li.style.marginLeft = `${(parseInt(header.tagName[1]) - 1) * 10}px`;

        // Crear enlace
        const a = document.createElement("a");
        a.href = `#${id}`;
        a.textContent = header.textContent;

        // Agregar enlace a la lista
        li.appendChild(a);
        ul.appendChild(li);
    });

    tocContainer.appendChild(ul);
});