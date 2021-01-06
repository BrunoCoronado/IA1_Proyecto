import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http'

@Injectable({
  providedIn: 'root'
})
export class WebService {
  protocolo: string = 'http://'
  host: string = '127.0.0.1:5000'
  path_raiz: string = ''

  constructor(private http: HttpClient) { }

  async consultar(data){
    return this.http.post(`${this.protocolo}${this.host}${this.path_raiz}/consultar`, data).toPromise();
  }

  async catalogos(){
    return this.http.get(`${this.protocolo}${this.host}${this.path_raiz}/catalogos`).toPromise();
  }

  async hiperparametros(){
    return this.http.get(`${this.protocolo}${this.host}${this.path_raiz}/hiperparametros`).toPromise();
  }
}
