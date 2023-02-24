const form = document.querySelector('form');
const table = document.querySelector('#recetas tbody');



form.addEventListener('submit', async function (event) {
	event.preventDefault();

    // const ingredientes = document.querySelector('#ingredientes').value;
    const ingredientes = document.getElementById('tbody-ingredientes');
	const similitud = document.querySelector('#similitud').value;
	const puntaje = document.querySelector('#puntaje').value;
	const precio = document.querySelector('#precio').value;

    const tabla = document.getElementById('recetas');
    const cargador = document.getElementById('cargando');
    const noresults = document.getElementById('no-results');

    const checkboxes = ingredientes.querySelectorAll('input[type="checkbox"]');
    console.log(checkboxes.length);
    // Inicializar variable para almacenar los ingredientes seleccionados
    let ingredientesSeleccionados = '';

    // Recorrer los checkboxes
    for(i=0;i<checkboxes.length;i++){
        const checkbox = checkboxes[i];       
        if (checkbox.checked) {
            // Agregar el valor del checkbox (que es el nombre del ingrediente) a la lista de ingredientes seleccionados
            ingredientesSeleccionados += String(checkbox.value) + ', ';
        }
    }


    // Eliminar la última coma y espacio
    ingredientesSeleccionados = ingredientesSeleccionados.slice(0, -2);
   
    noresults.classList.add('hidden');
    tabla.classList.add('hidden');
    cargador.classList.remove('hidden');


    url = 'http://localhost/filtrar_recetas';
    query = '?ingredientes=' + String(ingredientesSeleccionados) +
            '&similitud=' + String(similitud) +
            '&puntaje=' + String(puntaje) +
            '&precio=' + String(precio)
            ;

	try {
		const response = await fetch(url + query, {
			method: 'GET'
		});

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

	    let recetas = await response.json();
                
		table.innerHTML = '';

        cargador.classList.add('hidden');

        if  (recetas.length>0){
            noresults.classList.add('hidden');
            for(i=0;i<recetas.length;i++){
            receta = recetas[i]
                const row = table.insertRow();        
                row.insertCell().textContent = receta.nombre_del_platillo.slice(0,30);
                row.insertCell().textContent = receta.ingredientes.slice(0,100) + '...';
                row.insertCell().textContent = receta.puntaje_platillo;
                row.insertCell().textContent = receta.costo_receta;           
            }
            
            tabla.classList.remove('hidden');

        }else{
            tabla.classList.add('hidden');
            noresults.classList.remove('hidden');
        }
        

	} catch (error) {
        noresults.classList.remove('hidden');
        tabla.classList.add('hidden');
        cargador.classList.add('hidden');
		console.error('Error:', error);
	}
});
