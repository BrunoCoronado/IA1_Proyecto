import { Component, OnInit, ViewChild } from '@angular/core';
import { UtilsService } from './utils.service';
import { WebService } from './web.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'app';

  imagenes = []

  p_usac: number = 0;
  p_landivar: number = 0;
  p_mariano: number = 0;
  p_marroquin: number = 0;

  c_usac: number = 0;
  c_landivar: number = 0;
  c_mariano: number = 0;
  c_marroquin: number = 0;

  a_usac: number = 0;
  a_landivar: number = 0;
  a_mariano: number = 0;
  a_marroquin: number = 0;

  constructor(private service: WebService, private utils: UtilsService) { }

  async ngOnInit(): Promise<void> {

  }

  onUploadChange(event: any) {
    if(!event.srcElement.files[0].name.endsWith(".jpg")){
      (<HTMLInputElement>document.getElementById(`imagen`)).value = ""
      this.utils.mostrarError('Archivo invalido. Debe ser .jpg!')
      return
    }
    const nombre = (<HTMLInputElement>document.getElementById(`imagen`)).value.replace("C:\\fakepath\\", "");

    const file = event.target.files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
        let match =  reader.result.toString().match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);

        this.utils.mostrarLoading()
        const response: any = await this.service.analizar({ imagen: match[2] })
        this.utils.ocultarLoading()

        this.utils.mostrarToast(response.mensaje)

        let resultados = [
          { uni: 'USAC', porcentaje: response.resultados[0] },
          { uni: 'Landivar', porcentaje: response.resultados[1] },
          { uni: 'Mariano', porcentaje: response.resultados[2] },
          { uni: 'Marroquin', porcentaje: response.resultados[3] },
        ]

        resultados.sort((a, b) => b.porcentaje - a.porcentaje )
        
        this.imagenes.push({ img: reader.result.toString(), resultado: resultados[0].uni })

        if(nombre.includes('USAC')){
          this.c_usac++
          if(resultados[0].uni == 'USAC') this.a_usac++
          this.p_usac = (this.a_usac / this.c_usac) * 100

          // console.log(`p_usac ${this.p_usac}`)
          // console.log(`a_usac ${this.a_usac}`)
          // console.log(`c_usac ${this.c_usac}`)
        }else if(nombre.includes('Landivar')){
          this.c_landivar++
          if(resultados[0].uni == 'Landivar') this.a_landivar++
          this.p_landivar = (this.a_landivar / this.c_landivar) * 100

          // console.log(`p_landivar ${this.p_landivar}`)
          // console.log(`a_landivar ${this.a_landivar}`)
          // console.log(`c_landivar ${this.c_landivar}`)
        }else if(nombre.includes('Mariano')){
          this.c_mariano++
          if(resultados[0].uni == 'Mariano') this.a_mariano++
          this.p_mariano = (this.a_mariano / this.c_mariano) * 100

          // console.log(`p_mariano ${this.p_mariano}`)
          // console.log(`a_mariano ${this.a_mariano}`)
          // console.log(`c_mariano ${this.c_mariano}`)
        }else if(nombre.includes('Marroquin')){
          this.c_marroquin++
          if(resultados[0].uni == 'Marroquin') this.a_marroquin++
          this.p_marroquin = (this.a_marroquin / this.c_marroquin) * 100

          // console.log(`p_marroquin ${this.p_marroquin}`)
          // console.log(`a_marroquin ${this.a_marroquin}`)
          // console.log(`c_marroquin ${this.c_marroquin}`)
        }
    };
    (<HTMLInputElement>document.getElementById(`imagen`)).value = ""
  }
}
