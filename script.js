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
        // columnas de la respuesta: 
        // nombre_del_platillo, ingredientes, kcal, proteinas_gr, carbohidratos_gr, grasas_gr, puntaje_platillo, costo_receta
                
		table.innerHTML = '';

        cargador.classList.add('hidden');

        if  (recetas.length>0){
            noresults.classList.add('hidden');
            for(i=0;i<recetas.length;i++){
                const receta = recetas[i];
                const row = table.insertRow();   
                //row.setAttribute('onclick', 'showFichaReceta()');        
                
                //cell 0
                const platillo = row.insertCell();
                platillo.textContent = receta.nombre_del_platillo;
                platillo.id = "id_nombre_platillo";
                
                //cell 1
                row.insertCell().textContent = receta.ingredientes.slice(0,100) + '...';
                row.insertCell().textContent = receta.puntaje_platillo;

                //cell 2
                const costo_platillo = row.insertCell();
                costo_platillo.id = 'id_costo_platillo';
                costo_platillo.textContent = receta.costo_receta;

                //cell 3
                //ingredientes completos:
                const hidIng = row.insertCell();
                hidIng.id = 'id_ingredientes_platillo';
                hidIng.textContent = receta.ingredientes;
                hidIng.style.display = 'none';

                //info nutricional:
                //cell 4            
                const kcal = row.insertCell();
                kcal.id = 'id_kcal_platillo';
                //cell 5
                const prot = row.insertCell();
                prot.id = 'id_prot_platillo';
                //cell 6
                const carb = row.insertCell();
                carb.id = 'id_carb_platillo';
                //cell 7
                const grasa = row.insertCell();
                grasa.id = 'id_grasa_platillo';

                kcal.textContent = receta.kcal;
                kcal.style.display = 'none';
                prot.textContent = receta.proteinas_gr;
                prot.style.display = 'none';
                carb.textContent = receta.carbohidratos_gr;
                carb.style.display = 'none';
                grasa.textContent = receta.grasas_gr;
                grasa.style.display = 'none';

                row.onclick = function() {
                    showFichaReceta(receta.nombre_del_platillo, receta.costo_receta, receta.ingredientes, receta.kcal, receta.proteinas_gr, receta.carbohidratos_gr, receta.grasas_gr);
                  };

                         
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


  
  