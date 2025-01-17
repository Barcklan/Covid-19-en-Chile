// script.js
 // Seleccionar elementos
 const modal = document.getElementById("modal");
 const iframe = document.getElementById("modal-iframe");
 const modalTitle = document.getElementById("modal-title");
 const modalParagraph = document.getElementById("modal-paragraph");
 const closeModal = document.getElementById("close-modal");

 // Botones
 const btnAnimacion = document.getElementById("btn-animacion");
 //const btnTableau = document.getElementById("btn-tableau");

 // Abrir modal para animación
 btnAnimacion.addEventListener("click", () => {
    iframe.src = "animacion_muertes_covid_chile.html"; // Ruta del archivo de animación
    modalTitle.textContent = "Animación de Muertes por COVID-19";
    modalParagraph.textContent = "Este gráfico animado muestra las muertes acumulativas por COVID-19 en Chile.";
    modal.style.display = "flex";
    document.body.classList.add("modal-open"); // Añadir clase para deshabilitar transiciones
});
// Abrir modal para Tableau
//btnTableau.addEventListener("click", () => {
  //  iframe.src = "https://public.tableau.com/app/profile/barcklan.inforadd/viz/Covid-19_17366569215180/Covid19-P13#1"; // Ruta de Tableau
   // modalTitle.textContent = "Dashboard de Tableau: Muertes por COVID-19";
    //modalParagraph.textContent = "Este dashboard interactivo de Tableau presenta datos sobre las muertes por COVID-19 en Chile.";
    //modal.style.display = "flex";
//});
// Cerrar modal
closeModal.addEventListener("click", () => {
    modal.style.display = "none";
    iframe.src = ""; // Limpiar el iframe al cerrar
});
// Cerrar modal al hacer clic fuera del contenido
window.addEventListener("click", (event) => {
    if (event.target === modal) {
        modal.style.display = "none";
        iframe.src = ""; // Limpiar el iframe
        document.body.classList.remove("modal-open"); // Restaurar transiciones
    }
});
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
// Función para regresar al inicio de la página
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth' // Desplazamiento suave
    });
}
//res.cookie("nombreCookie", "valor", {
  //  sameSite: "Strict", // O "Lax", según el caso
   // secure: true // Requiere HTTPS
//});
document.cookie = "nombreCookie=valor; SameSite=Strict; Secure";
window.addEventListener('resize', function() {
    var iframe = document.querySelector("iframe");
    if (iframe && iframe.contentWindow.Plotly) {
        var plotDiv = iframe.contentDocument.querySelector('.plotly-graph-div');
        if (plotDiv) {
            iframe.contentWindow.Plotly.Plots.resize(plotDiv);
        }
    }
});
