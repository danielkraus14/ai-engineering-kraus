# Test questions for the RAG system (NOT indexed: this file lives outside data/)

These are the fixed questions used to verify the system. The two trap questions are in the
real estate domain but their answers are deliberately NOT in any file under data/.
Verified absent with grep: "cripto", "bitcoin", "mudanza".

## Questions answered in the documents

1. Question: "Cuanto es el deposito en garantia de un alquiler y se puede pagar en cuotas?"
   Expected: one month of rent, payable in up to 3 equal monthly installments, no interest.
   Source: data/alquileres.md (section 3).

2. Question: "Cuantos dormitorios tiene el departamento DV-101 y cuanto son las expensas?"
   Expected: 2 bedrooms, expenses of 62,000 pesos per month (price 78,000 USD).
   Source: data/propiedades.md (code DV-101).

3. Question: "Que porcentaje de honorarios paga el comprador y en que momento se pagan?"
   Expected: 3 percent of the sale price plus VAT, paid when the purchase agreement (boleto) is signed.
   Source: data/compraventa.md (stage 5).

## Trap questions (answer NOT in the documents)

4. Question: "Se pueden pagar los honorarios o el alquiler con criptomonedas?"
   Expected: the system says it does not know / has no information. It must not invent a policy.

5. Question: "La inmobiliaria ofrece servicio de mudanza para los inquilinos nuevos?"
   Expected: the system says it does not know / has no information. It must not invent a service.
