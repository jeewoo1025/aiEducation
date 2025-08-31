# Typescript

- 핸드북: https://joshua1988.github.io/ts/why-ts.html
- 실습코드: https://github.com/joshua1988/learn-typescript

## 기본 
- 마이크로소프트에서 개발한 오픈소스 언어
- 장점
  - 에러의 사전 방지: 실제로 코드 실행 전, 미리 에러를 검출 가능
  - 코드 가이드 및 자동완성: VSCode에서 빠른 개발 지원 가능

## 개발환경 구축
- [ [Typescript] 개발 환경 구축 ](https://tango1202.github.io/typescript/typescript-config/)

## 프로젝트 폴더 구조
- node_modules: 실습에 필요한 라이브러리 설치되어있음. 
- src
  - package-lock.json: dependencies나 devDependencies에 명시된 라이브러리를 설치할 때 필요한 부수 라이브러리의 버전을 관리함. (자동 생성. 개발자가 직접 관리 X)
  - package.json: NPM 설정 파일. 프로젝트 이름, 버전, 라이선스 등 프로젝트와 관련된 기본 정보가 들어감. 
  - tsconfig.json: typescript 설정 파일. 타입스크립트 컴파일을 돌릴 파일 목록, 배제할 목록, 컴파일러에 대한 구체적인 동작 옵션 지정 가능.


### 기본 타입
```typescript
// 기본 타입에 맞는 예시들

// string: 문자열 타입
const userName: string = "captain";
const greeting: string = "Hello, World!";

// number: 숫자 타입
const userAge: number = 25;
const price: number = 19.99;
const count: number = 100;

// boolean: 논리 타입
const isActive: boolean = true;
const hasPermission: boolean = false;

// object: 객체 타입
const user: object = { 
    name: "captain", 
    age: 25,
    isActive: true 
};

// Array: 배열 타입
const hobbies: string[] = ["reading", "swimming", "coding"];
const numbers: number[] = [1, 2, 3, 4, 5];
const mixedArray: (string | number)[] = ["hello", 123, "world"];

// tuple: 고정된 길이와 타입의 배열
const address: [string, number] = ["Seoul", 1000];
const coordinates: [number, number] = [10.5, 20.3];

// any: 모든 타입 허용
const dynamicValue: any = "hello";
dynamicValue = 123;
dynamicValue = true;
dynamicValue = { name: "captain" };

// null: null 값
const nullableValue: null = null;

// undefined: undefined 값
const undefinedValue: undefined = undefined;
```

### 함수
```typescript
// TypeScript 함수 인자 예시

// 1. 기본 인자 (Required Parameter)
function greet(name: string): string {
    return `Hello, ${name}!`;
}

// 2. 선택적 인자 (Optional Parameter)
function greetWithAge(name: string, age?: number): string {
    if (age) {
        return `Hello, ${name}! You are ${age} years old.`;
    }
    return `Hello, ${name}!`;
}

// 3. 기본값 인자 (Default Parameter)
function createProfile(name: string, age: number = 18): string {
    return `${name} is ${age} years old.`;
}

// 4. 나머지 인자 (Rest Parameter)
function sum(...numbers: number[]): number {
    return numbers.reduce((total, num) => total + num, 0);
}

// 5. 객체 인자 (Object Parameter)
function createUser(user: { name: string; age: number; email?: string }): string {
    return `User: ${user.name}, Age: ${user.age}`;
}

// 6. 인터페이스를 사용한 객체 인자
interface User {
    name: string;
    age: number;
    email?: string;
}

function updateUser(user: User): string {
    return `Updated: ${user.name}`;
}

// 7. 배열 인자
function processHobbies(hobbies: string[]): void {
    hobbies.forEach(hobby => console.log(hobby));
}

// 8. 튜플 인자
function setLocation(coords: [number, number]): void {
    console.log(`Latitude: ${coords[0]}, Longitude: ${coords[1]}`);
}

// 9. 유니온 타입 인자
function handleValue(value: string | number): void {
    console.log(value.toString());
}

// 10. any 타입 인자
function processData(data: any): any {
    return data;
}

// 사용 예시
greet("captain");                           // "Hello, captain!"
greetWithAge("captain", 25);                // "Hello, captain! You are 25 years old."
greetWithAge("captain");                   // "Hello, captain!"
createProfile("captain");                  // "captain is 18 years old."
createProfile("captain", 30);              // "captain is 30 years old."
sum(1, 2, 3, 4, 5);                         // 15
createUser({ name: "captain", age: 25 });  // "User: captain, Age: 25"
processHobbies(["reading", "swimming"]);  // 각 취미 출력
setLocation([37.5665, 126.9780]);          // 좌표 출력
handleValue("hello");                      // "hello"
handleValue(123);                          // "123"
```

### 인터페이스
- 객체 타입을 정의할 때 사용하는 문법 
- `extends` 키워드를 통해 상속이 가능하며 하나의 인터페이스는 여러 인터페이스를 동시에 상속할 수 있는 다중상속도 지원함
```typescript
// 기본 인터페이스 정의
interface Person {
  name: string;
  age: number;
  email?: string; // 선택적 프로퍼티
}

// 인터페이스 상속 예시
interface Employee extends Person {
  employeeId: string;
  department: string;
}

// 다중 인터페이스 상속 예시
interface Flyable {
  fly(): void;
}

interface Bird extends Person, Flyable {
  canFly: boolean;
  layEggs(): void;
}

// 함수 파라미터로 사용하는 인터페이스
interface Product {
  id: number;
  name: string;
  price: number;
}

function processProduct(product: Product): void {
  console.log(`Processing ${product.name} with price ${product.price}`);
}

// 클래스 구현 예시
interface Drawable {
  draw(): void;
}

class Circle implements Drawable {
  draw() {
    console.log("Drawing a circle");
  }
}
```

### 타입 별칭 (type alias)
- 특정 타입이나 인터페이스 등을 참조할 수 있는 타입 변수; 즉, 타입에 의미를 부여해서 별도의 이름을 부르는 것
- 










